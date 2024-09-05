## Environments Preparation
Step 1: apply a patch to `nuscenes-devkit` for env support
```
cd dependencies/nuscenes-devkit
git apply --exclude='*.DS_Store' ../../patch/nuscenes-devkit.patch
cp -r ./python-sdk/nuscenes ../../docker
```
Step 2: build docker image
```
cd ../../docker
docker build -t uniad_torch1.12 -f uniad_torch1.12.dockerfile .
```

<- Last Page: [Project Installation](proj_setup.md)

-> Next Page: [Data Preparation](data_prep.md)
