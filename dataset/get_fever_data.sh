mkdir ~/git/acl2020-confidence-regularization/dataset/fever
cd ~/git/acl2020-confidence-regularization/dataset/fever
wget https://www.dropbox.com/s/bdwf46sa2gcuf6j/fever.dev.jsonl
wget https://www.dropbox.com/s/v1a0depfg7jp90f/fever.train.jsonl
cd ~/git
git clone https://github.com/TalSchuster/FeverSymmetric
cp ./FeverSymmetric/symmetric_v0.2/* ~/git/acl2020-confidence-regularization/dataset/fever/
cp ./FeverSymmetric/symmetric_v0.1/* ~/git/acl2020-confidence-regularization/dataset/fever/
