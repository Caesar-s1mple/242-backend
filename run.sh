export LANG="en_US.UTF-8"

chmod +x run.sh
python inference.py &
python dailydata_socket.py &
python api.py &
wait
