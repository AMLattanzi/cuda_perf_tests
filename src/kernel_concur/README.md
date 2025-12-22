# kernel_concur

The following code demonstrates the benefit of concurrency with 2 streams. The operations of copying
host-to-device, kernel execution, and copying device-to-host, are timed with and without streams.
Since each stream executes the all of the work (H2D,kernel,D2H), but on a subset of the data, the
work may be staggered as follows:
```
==============================================
|   H2D   |         Kernel        |    D2H   | (Total)
==============================================   
======================
| H2D | Kernel | D2H |        (Stream 0)
======================
       ======================
       | H2D | Kernel | D2H | (Stream 1)
       ======================
```


The following output is obtained on a single A100 GPU on the NERSC Perlmutter machine.
```
No concurrency compute time (ms): 70
With concurrency compute time (ms): 37
```