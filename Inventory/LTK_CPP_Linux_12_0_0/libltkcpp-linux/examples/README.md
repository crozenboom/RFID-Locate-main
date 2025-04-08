# Impinj Example Application

## Overview

This example illustrates using CMake to build an Impinj Reader application for
host (i.e. x86\_64) and Impinj Reader CPU architectures.  The example
application uses Impinj LTK libraries that are found inside the ETK.  The
'build.sh' script is intended to be run from within the Impinj ETK.

The ETK contains:
  * a cross compilation toolchain that targets an Impinj Reader architecture
  * libraries built for an Impinj Reader architecture
  * host architecture versions of libraries to allow host builds for faster
    development and debugging)

## Building

To build the project, call the build script from the command line:

```bash
[user@machine]$ ./build.sh
```

The script accepts overrides for certain variables.  See 'build.sh' for more
information.

## Running

To run on the host machine, simply run the exectuable:

```bash
[user@machine]$ output/host/docsample1/docsample1 ${READER_HOSTNAME}
```

To run on an Impinj Reader, e.g. an R700, copy the binary to the Reader and
execute it:

```bash
[user@machine]$ ssh root@r700
Password: ********
>
  
# drop in to osshell
> osshell <code>
    
# copy the binary and execute it
root@r700:~# scp user@machine:/path/to/output/target/docsample1/docsample1 /tmp
root@r700:~# /tmp/docsample1 localhost
```
