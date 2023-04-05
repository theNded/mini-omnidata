##!/usr/bin/env bash

pip install gdown
mkdir -p pretrained_models

gdown '1Jrh-bRnJEjyMCS7f-WsaFlccfPjJPPHI&confirm=t' -O ./pretrained_models/ # omnidata depth (v2)
gdown '1wNxVO4vVbDEMEpnAi_jwQObf2MFodcBR&confirm=t' -O ./pretrained_models/ # omnidata normals (v2)
