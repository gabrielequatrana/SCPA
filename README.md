# Progetto SCPA
Questa repository contiene il codice e la relazione per il progetto del corso di SCPA.

- La cartella ```code``` contiene il codice del programma.
- La cartella ```data``` contiene i dati utilizzati dal programma.

## Guida all'uso
Per compilare il programma:
- Eseguire il comando `cmake .`
- Eseguire il comando `make`

È necessario inserire nella cartella `data/matrix` i file `.mtx` relativi alle matrici da utilizzare:
- È possibile usare lo script bash `download_matrix.sh` presente nella cartella.

Per eseguire il programma in modalità CPU usare il comando:
```
./spmmCPU [ -csr | -ell ]
```
Per eseguire il programma in modalità GPU usare  il comando:
```
./spmmGPU [ -csr | -ell ]
```
I risultati vengono salvati nella cartella ```data/csv```
