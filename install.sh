# install boost C++ dependencies
sudo apt install libboost-dev

# install MongoDB
curl -fsSL https://pgp.mongodb.com/server-6.0.asc | sudo gpg -o /usr/share/keyrings/mongodb-server-6.0.gpg --dearmor

# install graph tools for evaluation
conda install -c conda-forge -c ostrokach-forge -c pkgw-forge graph-tool

# initialize a MongoDB server
screen 
mongod
