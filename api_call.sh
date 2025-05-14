#!/bin/bash

patient_id="6gx2vgpSvj4CmUeQtqg5"

if [ "$1" == "online" ]; then
	echo "setting patient status to online"
	curl -X PUT "https://fypbackend-7262ace852de.herokuapp.com/api/patients/$patient_id/online" \
	-H "Content-Type: application/json" \
	-d '{}'
	echo ""

elif [ "$1" == "notification" ]; then
	echo "Patient has entered pre-ictal state. Sending Notification..."
	curl -X POST https://fypbackend-7262ace852de.herokuapp.com/api/simulate_prediction \
	-H "Content-Type: application/json" \
	-d "{\"patient_id\":\"$patient_id\"}"

elif [ "$1" == "offline" ]; then
	echo "setting patient status offline"
	curl -X PUT "https://fypbackend-7262ace852de.herokuapp.com/api/patients/$patient_id/offline" \
	-H "Content-Type: application/json" \
	-d '{}'

elif [ "$1" == "image" ]; then
	echo "Sending image"
	curl -X POST "https://fypbackend-7262ace852de.herokuapp.com/api/upload_image" \
	-F "patient_id=$patient_id" \
	-F "image=@/home/nvidia/Documents/inference/BNT/binary_image.jpg"
fi
