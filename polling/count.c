/* POLLING */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

/* Definitions */
#define MAX_QUEST 5
#define FILENAME "./votations.dat"

typedef struct {
     int idTable;
     int idQuestion;
     int yes;
     int no;
  }  tRecord;

/* Main program */
int main (int argc, char **argv)
{
   int    rank, nprocs;
   int    recordsize, filenumrecords, rem, numrecords;
   int    i, quest;
   tRecord *buf;
   long  total, yes[MAX_QUEST], totYes[MAX_QUEST];
   long  no[MAX_QUEST], totNo[MAX_QUEST];

   MPI_Status   info;
   MPI_Datatype rectype; 
   MPI_File     infile;
   MPI_Offset   filesize, poffset;

   /* MPI Initialization */
   MPI_Init(&argc, &argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

   /* Create datatype */
    MPI_Datatype types[4] = {MPI_INT, MPI_INT, MPI_INT, MPI_INT};
    int blockcounts[4] = {1, 1, 1, 1};
    MPI_Aint offsets[4];
    offsets[0] = offsetof(tRecord, idTable);
    offsets[1] = offsetof(tRecord, idQuestion);
    offsets[2] = offsetof(tRecord, yes);
    offsets[3] = offsetof(tRecord, no);
    MPI_Type_create_struct(4, blockcounts, offsets, types, &rectype);
    MPI_Type_commit(&rectype);
    
   /* Each process reads a part of the file */
    MPI_File_open(MPI_COMM_WORLD, FILENAME, MPI_MODE_RDONLY, MPI_INFO_NULL, &infile);

   /* Each process determines number of records to read and initial offset */
    MPI_File_get_size(infile, &filesize);
    
    recordsize = sizeof(tRecord);
    filenumrecords = filesize / recordsize;
    rem = filesize % recordsize;
    numrecords = filenumrecords / nprocs;

    if (rank == nprocs - 1) numrecords += rem / recordsize;

    poffset = rank * numrecords * recordsize;
    
   /* Allocate buffer for records */
    buf = (tRecord *)malloc(numrecords * recordsize);
    
   /* Process reads numrecords consecutive elements */
    MPI_File_set_view(infile, poffset, rectype, rectype, "native", MPI_INFO_NULL);
    MPI_File_read(infile, buf, numrecords, rectype, &info);
    
   /* Count results by each process */
    total = 0;
    for (quest = 0; quest < MAX_QUEST; quest++) {
        yes[quest] = 0;
        no[quest] = 0;
    }

    for (i = 0; i < numrecords; i++) {
        for (quest = 0; quest < MAX_QUEST; quest++) {
            if (buf[i].idQuestion == quest) {
                yes[quest] += buf[i].yes;
                no[quest] += buf[i].no;
                total += buf[i].yes + buf[i].no;
            }
        }
    }
    
   /* Print local results */
   // To print the local results in order we'll pass all the results to proc. 0 and then print it out
    long *all_proc_votes = NULL;
    if (rank == 0) {
        all_proc_votes = (long *)malloc(nprocs * sizeof(long));
    }
    MPI_Gather(&total, 1, MPI_LONG, all_proc_votes, 1, MPI_LONG, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        for (i = 0; i < nprocs; i++) {
            printf("Proc %3d. Counted votes = %ld\n", i, all_proc_votes[i]);
        }
    }

   /* Print global results on process 0 */
    MPI_Reduce(yes, totYes, MAX_QUEST, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(no, totNo, MAX_QUEST, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    
   if (rank == 0)
   {
      total = 0;
      printf ("------------------------------------------------------------\n");
      for (i=0; i<MAX_QUEST; i++) 
      {
         total += totYes[i] + totNo[i];
         printf("Question %d: yes: %.1f%% (%d) no: %.1f%% (%d)\n", i, totYes[i]*100.0/total, totYes[i], totNo[i]*100.0/total, totNo[i]);
         fflush (stdout);
      }

      printf ("------------------------------------------------------------\n");
      printf ("Total votes = %d\n", total);
      fflush (stdout);
   }

   /* Free datatype */
   MPI_Type_free (&rectype);

   /* End MPI */
   MPI_Finalize();
   return 0;
}

