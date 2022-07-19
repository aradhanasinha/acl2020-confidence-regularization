CMD=$1
echo $CMD
until $CMD; do
	echo "*************** Job crashed. Respawning... ***************************"
	sleep 1
done
