#!/usr/bin/bash

POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
	case $1 in
		-r | --run)
		RUN=1
		shift
		;;
		-* | --*)
		echo "Unknown optional arguments"
		exit 1
		;;
		*)
		
		POSITIONAL_ARGS+=("$1") # save positional arg
      	shift # past argument
      	;;
	esac
done

set -- "${POSITIONAL_ARGS[@]}" # restore positional parameters
	
source="$1"

g++ $source -o test -I/usr/local/include/opencv4 -lopencv_highgui -lopencv_core -lopencv_imgcodecs -lopencv_imgproc -lopencv_features2d -lopencv_xfeatures2d -lopencv_calib3d

if [[ $? -eq 0 ]] && [[ $RUN ]]; then
	./test
fi
