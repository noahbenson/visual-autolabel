#! /bin/bash

RESOURCE_GROUP_NAME="visual-autolabel"
STORAGE_ACCOUNT_NAME="nbenvisualautolabel"
FILE_SHARE_NAME="nbenvisualautolabel-fileshare"
BLOB_PATH="/blob"
DATA_PATH="/data"

mkdir -p "$DATA_PATH"
mkdir -p "$BLOB_PATH"
chown azureuser.azureuser "$DATA_PATH"
chown azureuser.azureuser "$BLOB_PATH"

[ "`whoami`" = "root" ] || {
    echo "Init script must be run as root!"
    exit 1
}

# Install packages via apt:
apt update && apt install --yes azure-cli docker.io

# Make sure that the azureuser can use docker.
adduser azureuser docker

# For most of the remaining commands, we'll need to login to the azure system.
az account show &>/dev/null || az login || {
    echo "The command `az login` failed."
    exit 2
}

# Extract the vm-data and put it in the root.
curl -s -H Metadata:true \
     "http://169.254.169.254/metadata/instance/compute?api-version=2017-08-01" \
     > /vmdata.json
az tag list > /vmtags.json
chmod 644 /vmdata.json
chmod 644 /vmtags.json
python3 <<EOF
import json
with open('/vmdata.json', 'rt') as f:
    data = json.load(f)
with open('/vmtags.json', 'rt') as f:
    tags = json.load(f)
vmid = data['vmId']
vmname = data['name']
try:
    (start, stop, stride) = [
        next(el['values'][0]['tagValue'] for el in tags if el['tagName'] == k)
        for k in ('start','stop','stride')]
except Exception:
    (start, stop, stride) = ('', '', '')
# If the vmname matches a specific pattern
with open('/vmconfig.sh', 'wt') as f:
    print(f'VM_ID="{vmid}"', file=f)
    print(f'VM_NAME="{vmname}"', file=f)
    print(f'VM_START="{start}"', file=f)
    print(f'VM_STOP="{stop}"', file=f)
    print(f'VM_STRIDE="{stride}"', file=f)
EOF
# Store these for sourcing by users.
echo "DATA_PATH=\"${DATA_PATH}\"" >> /vmconfig.sh
echo "BLOB_PATH=\"${BLOB_PATH}\"" >> /vmconfig.sh
chmod 644 /vmconfig.sh

# Go ahead and get the blob path mounted
HTTP_ENDPOINT=$(az storage account show \
		   --resource-group $RESOURCE_GROUP_NAME \
		   --name $STORAGE_ACCOUNT_NAME \
		   --query "primaryEndpoints.file" --output tsv | tr -d '"')
SMB_PATH=$(echo $HTTP_ENDPOINT | cut -c7-${#HTTP_ENDPOINT})$FILE_SHARE_NAME

STORAGE_ACCOUNT_KEY=$(az storage account keys list \
			 --resource-group $RESOURCE_GROUP_NAME \
			 --account-name $STORAGE_ACCOUNT_NAME \
			 --query "[0].value" --output tsv | tr -d '"')

mount -t cifs "$SMB_PATH" "$BLOB_PATH" \
      -o username=$STORAGE_ACCOUNT_NAME,password=$STORAGE_ACCOUNT_KEY,serverino,nosharesock,actimeo=30,mfsymlinks,rw,uid=1000,gid=1000

# That's everything for the initialization.
echo ""
echo "Initialization successful."
exit 0
