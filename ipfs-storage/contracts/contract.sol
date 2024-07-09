// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract IPFSStorage {
    string public cid;

    function setCID(string memory _cid) public {
        cid = _cid;
    }

    function getCID() public view returns (string memory) {
        return cid;
    }
}
