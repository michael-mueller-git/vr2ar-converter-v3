# HereSphere 

## Disable Guardian 

Load required tools:

```sh
nix-shell -p apksigner apktool zulu8
```

1. Extract App:

```sh
adb devices  # check if device is detected
adb shell pm list packages # find package
adb pull "$(adb shell pm path com.heresphere.vrvideoplayer | cut -d ':' -f2)"
apktool d base.apk
```

Open `AndroidManifest.xml` and find line: 

```xml
<uses-feature android:name="com.oculus.feature.PASSTHROUGH" android:required="true"/>
```

insert the following below this line:

```xml
<uses-feature android:name="com.oculus.feature.BOUNDARYLESS_APP" android:required="true" />
```

then compile the apk again:

```sh
apktool b base
keytool -genkey -v -keystore keyStore.keystore -alias app -keyalg RSA -keysize 2048 -validity 10000
apksigner sign --ks keyStore.keystore base/dist/base.apk
```

finaly uppload:

```sh
adb devices  # check if device is detected
# 1. remove original app because andorind does not allow installing an update if signature do not match
adb uninstall com.heresphere.vrvideoplayer
# 2. install patched apk
adb install -g -r base/dist/base.apk
```

