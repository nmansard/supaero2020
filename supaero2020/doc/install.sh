sudo apt install curl aptitude git adb
sudo sh -c "echo 'deb [arch=amd64] http://robotpkg.openrobots.org/packages/debian/pub xenial robotpkg' >> /etc/apt/sources.list.d/robotpkg.list"
curl http://robotpkg.openrobots.org/packages/debian/robotpkg.key | sudo apt-key add -
sudo apt-get update
sudo apt install robotpkg-py27-pinocchio robotpkg-gepetto-viewer-corba ipython ipython3 python-pip python3-pip freeglut3 python-matplotlib python3-matplotlib python3-pil.imagetk
pip3 install --user 'cozmo[camera,3dviewer]'
pip3 install --user tflearn tensorflow
pip install --user tflearn tensorflow
pip3 install --user ipython


echo '
export PATH=/opt/openrobots/bin:$PATH
export PKG_CONFIG_PATH=/opt/openrobots/lib/pkgconfig:$PKG_CONFIG_PATH
export LD_LIBRARY_PATH=/opt/openrobots/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/opt/openrobots/lib/python2.7/site-packages:$PYTHONPATH
' >> ~/.bashrc
source ~/.bashrc

export WITH_PYOMO=

if [ ! -z $WITH_PYOMO ] ; then
  cd /tmp   
  wget https://repo.anaconda.com/archive/Anaconda3-5.3.1-Linux-x86_64.sh
  sh Anaconda3-5.3.1-Linux-x86_64.sh
  echo '
  export PATH=/home/student/anaconda3/bin:$PATH
  export LD_LIBRARY_PATH=/home/student/anaconda3/lib:$LD_LIBRARY_PATH
  export PYTHONPATH=/home/student/anaconda3/lib/python2.7/site-packages:$PYTHONPATH
  ' >> ~/.bashrc
  source ~/.bashrc
  conda install -c conda-forge pyomo
  #conda install -c conda-forge pyomo.extra
  conda install -c conda-forge ipopt
fi 

