const IPFSStorage = artifacts.require("IPFSStorage");

module.exports = function (deployer) {
    deployer.deploy(IPFSStorage);
};
