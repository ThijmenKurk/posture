killall droidcam-cli
droidcam-cli -l 4747 &
disown -h %1
ssh -p 2222 192.168.2.187 ./connect_droidcam.sh

# contents of './connect_droidcam.sh' on Android phone:
# am force-stop com.dev47apps.droidcamx
# am start -n com.dev47apps.droidcamx/com.dev47apps.droidcamx.DroidCamX -a android.intent.action.MAIN -c android.intent.category.LAUNCHER

# input keyevent 82
# input keyevent 66
# input keyevent 66
# input keyevent 66

