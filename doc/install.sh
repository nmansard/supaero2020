sudo apt install curl aptitude git adb
sudo sh -c "echo 'deb [arch=amd64] http://robotpkg.openrobots.org/packages/debian/pub bionic robotpkg' >> /etc/apt/sources.list.d/robotpkg.list"
sudo sh -c "echo 'deb [arch=amd64] http://robotpkg.openrobots.org/wip/packages/debian/pub bionic robotpkg' >> /etc/apt/sources.list.d/robotpkg.list"
curl http://robotpkg.openrobots.org/packages/debian/robotpkg.key | sudo apt-key add -
sudo apt-get update
sudo apt install robotpkg-py3\*-pinocchio robotpkg-py3\*-qt4-gepetto-viewer-corba ipython3 python3-pip freeglut3 python3-matplotlib python3-pil.imagetk jupyter python3-scipy python3-matplotlib jupyter python3-scipy python3-numpy robotpkg-example-robot-data robotpkg-osg-dae


echo '
export PATH=/opt/openrobots/bin:$PATH
export PKG_CONFIG_PATH=/opt/openrobots/lib/pkgconfig:$PKG_CONFIG_PATH
export LD_LIBRARY_PATH=/opt/openrobots/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/opt/openrobots/lib/python3.6/site-packages:$PYTHONPATH
' >> ~/.bashrc
source ~/.bashrc

