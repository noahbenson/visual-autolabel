#! /bin/bash
###############################################################################
# Startup script for the Visual Autolabel docker image.
#
# This script gets run whenever the visual-autolabel docker image is run using
# the `docker run` command.
#
# The default behavior is to start a jupyter server on port 8888, which can
# also be achieved using the "jupyter" sub-command, i.e.:
#    docker run -it nben/visual_autolabel:latest jupyter
# Other behaviors can be achieved with subcommands that are documented in the
# help function below.


# Functions ###################################################################

function help_msg {
    echo "SYNTAX: docker run nben/visual_autolabel:latest <command> <args...>"
    echo ""
    echo "The following subcommands are recognized:"
    echo " * jupyter"
    echo "   Starts a jupyter server on port 8888; use with the -p8888:8888"
    echo "   option (to the docker run command) then point your web-browser to"
    echo "   localhost:8888 to connect to this server."
    echo " * bash"
    echo "   Starts a BASH session; should be used with the interactive (-it)"
    echo "   docker run option."
    echo " * apply | generate | train"
    echo "   Runs the command as a visual_autolabel.cmd library command. These"
    echo "   commands either apply visual_autolabel model to subjects,"
    echo "   generate data used by Benson, Song, et al. (2025), or train one"
    echo "   of the models trained in the same paper."
    echo ""
    echo "For additional help, run one of the above subcommands with the"
    echo "argument help."
}
function syntax {
    help_msg
    exit 1
}
function error {
    echo "$@" 1>&2
}
function die {
    error "$@"
    exit 1
}


# Script ######################################################################

# Process the arguments:
[ -z "$1" ] && syntax
case "$1" in
    help)
        help_msg
        exit 0
        ;;
    jupyter)
        exec /usr/local/bin/start-notebook.sh \
             --no-browser \
             --port 8888 \
             --ip='*' --NotebookApp.token='' --NotebookApp.password=''
        ;;
    bash)
        exec /bin/bash
        ;;
    *)
        # Anything else is a command for the visual autolabel library.
        exec python -m visual_autolabel.cmd "$@"
        ;;
esac

# We shouldn't ever reach this point.
exit 1
