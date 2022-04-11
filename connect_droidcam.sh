killall droidcam-cli
droidcam-cli -l 4747 &
disown -h %1
ssh -p 2222 192.168.2.187 ./connect_droidcam.sh
