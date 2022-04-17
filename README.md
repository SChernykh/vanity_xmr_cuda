# Monero vanity address generator

Usage:
```
./vanity_xmr_cuda [-i] pattern1 [pattern_2] [pattern_3] ... [pattern_n]
```

`-i` makes the search case insensitive. You can't use capital letters in patterns in `-i` mode. Each pattern can have `?` symbols which match any character.

Example:
```
./vanity_xmr_cuda -i 4?xxxxx 433333 455555 477777 499999
```

If the vanity generator finds a match, it will print the spend secret key and the resulting Monero address. Copy the spend key and run `./monero-wallet-cli --generate-from-spend-key WALLET_NAME` to create this wallet. Paste the spend key when the wallet asks you for it.

## Performance

This generator can check ~8.1 million keys/second on GeForce GTX 1660 Ti. Multiple patterns don't slow down the search.

## Build instructions

### Ubuntu

Run the following commands to install the necessary prerequisites, clone this repo, and build P2Pool locally on Ubuntu 20.04:
```
git clone https://github.com/SChernykh/vanity_xmr_cuda
cd vanity_xmr_cuda
mkdir build && cd build
cmake ..
make -j$(nproc)
```

## Donations

If you'd like to support further development of this software, you're welcome to send any amount of XMR to the following address:

```
44MnN1f3Eto8DZYUWuE5XZNUtE3vcRzt2j6PzqWpPau34e6Cf4fAxt6X2MBmrm6F9YMEiMNjN6W4Shn4pLcfNAja621jwyg
```
