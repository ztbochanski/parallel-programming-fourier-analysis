// module  load  slurm 
// module  load  openmpi
// mpic++ fourier.cpp -o fourier  -lm
// mpiexec -np 4 fourier

#include <stdio.h>
#include <math.h>
#include <mpi.h>

#define F_2_PI			(float)(2.*M_PI)

// which node is in charge?

#define BOSS 0

// files to read and write:

#define BIGSIGNALFILEBIN	(char*)"bigsignal.bin"
#define BIGSIGNALFILEASCII	(char*)"bigsignal.txt"
#define CSVPLOTFILE		(char*)"plot.csv"

// tag to "scatter":

#define TAG_SCATTER		'S'

// tag to "gather":

#define TAG_GATHER		'G'

// how many elements are in the big signal:

#define NUMELEMENTS	(1*1024*1024)

// only consider this many periods (this is enough to uncover the secret sine waves):

#define MAXPERIODS	100

// which file type to read, BINARY or ASCII (BINARY is much faster to read):

#define BINARY

// print debugging messages?

#define DEBUG		true

// globals:

float * BigSums;		// the overall MAXPERIODS autocorrelation array
float *	BigSignal;		// the overall NUMELEMENTS-big signal data
int	NumCpus;		// total # of cpus involved
float * PPSums;			// per-processor autocorrelation sums
float *	PPSignal;		// per-processor local array to hold the sub-signal
int	PPSize;			// per-processor local array size

// function prototype:

void	DoOneLocalFourier( int );


int
main( int argc, char *argv[ ] )
{
	MPI_Status status;

	MPI_Init( &argc, &argv );

	int  me;		// which one I am

	MPI_Comm_size( MPI_COMM_WORLD, &NumCpus );
	MPI_Comm_rank( MPI_COMM_WORLD, &me );

	// decide how much data to send to each processor:

	PPSize    = NUMELEMENTS / NumCpus;

	// local arrays:

	PPSignal  = new float [PPSize];			// per-processor signal
	PPSums    = new float [MAXPERIODS];		// per-processor sums of the products

	// read the BigSignal array:

	if( me == BOSS )	// this is the big-data-owner
	{
		BigSignal = new float [NUMELEMENTS];		// so we can duplicate part of the array

#ifdef ASCII
		FILE *fp = fopen( BIGSIGNALFILEASCII, "r" );
		if( fp == NULL )
		{
			fprintf( stderr, "Cannot open data file '%s'\n", BIGSIGNALFILEASCII );
			return -1;
		}

		for( int i = 0; i < NUMELEMENTS; i++ )
		{
			float f;
			fscanf( fp, "%f", &f );
			BigSignal[i] = f;
		}
#endif

#ifdef BINARY
		FILE *fp = fopen( BIGSIGNALFILEBIN, "rb" );
		if( fp == NULL )
		{
			fprintf( stderr, "Cannot open data file '%s'\n", BIGSIGNALFILEBIN );
			return -1;
		}

		fread( BigSignal, sizeof(float), NUMELEMENTS, fp );
#endif
	}

	// create the array to hold all the sums:

	if( me == BOSS )
	{
		BigSums = new float [MAXPERIODS];	// where all the sums will go
	}

	// start the timer:

	double time0 = MPI_Wtime( );

	// have the BOSS send to itself (really not a "send", just a copy):

	if( me == BOSS )
	{
		for( int i = 0; i < PPSize; i++ )
		{
			PPSignal[i] = BigSignal[ BOSS*PPSize + i ];
		}
	}

	// getting the signal data distributed:

	if( me == BOSS )
	{
		// have the BOSS send to everyone else:
		for( int dst = 0; dst < NumCpus; dst++ )
		{
			if( dst == BOSS )
				continue;

			MPI_Send( &BigSignal[dst * PPSize], PPSize, MPI_FLOAT, dst, TAG_SCATTER, MPI_COMM_WORLD );
		}
	}
	else
	{
		// have everyone else receive from the BOSS:
		MPI_Recv( PPSignal, PPSize, MPI_FLOAT, BOSS, TAG_SCATTER, MPI_COMM_WORLD, &status );
	}

	// each processor does its own autocorrelation:

	DoOneLocalFourier( me );

	// get the sums back:

	if( me == BOSS )
	{
		// get the BOSS's sums:
		for( int s = 0; s < MAXPERIODS; s++ )
		{
			BigSums[s] = PPSums[s];		// start the overall sums with the BOSS's sums
		}
	}
	else
	{
		// each processor sends its sums back to the BOSS:
		MPI_Send( PPSums, MAXPERIODS, MPI_FLOAT, BOSS, TAG_GATHER, MPI_COMM_WORLD );
	}

	// the BOSS receives the sums and adds them into the overall sums:

	if( me == BOSS )
	{
		float tmpSums[MAXPERIODS];
		for( int src = 0; src < NumCpus; src++ )
		{
			if( src == BOSS )
				continue;

			// the BOSS receives everyone else's sums:
			MPI_Recv( tmpSums, MAXPERIODS, MPI_FLOAT, src, TAG_GATHER, MPI_COMM_WORLD, &status );
			for( int s = 0; s < MAXPERIODS; s++ )
				BigSums[s] += tmpSums[s];
		}
	}

	// stop the timer:

	double time1 = MPI_Wtime( );

	// print the performance:

	if( me == BOSS )
	{
		double seconds = time1 - time0;
                double performance = (double)NumCpus*(double)MAXPERIODS*(double)PPSize/seconds/1000000.;        // mega-mults computed per second
                fprintf( stderr, "%3d processors, %10d elements, %9.2lf mega-multiplies computed per second\n",
			NumCpus, NUMELEMENTS, performance );
	}

	// write the file to be plotted to look for the secret sine wave:

	if( me == BOSS )
	{
		FILE *fp = fopen( CSVPLOTFILE, "w" );
		if( fp == NULL )
		{
			fprintf( stderr, "Cannot write to plot file '%s'\n", CSVPLOTFILE );
		}
		else
		{
			for( int s = 1; s < MAXPERIODS; s++ )		// BigSums[0] is huge -- don't use it
			{
				fprintf( fp, "%6d , %10.2f\n", s, BigSums[s] );
			}
			fclose( fp );
		}
	}

	// all done:

	MPI_Finalize( );
	return 0;
}


// read from the per-processor signal array, write into the local sums array:

void
DoOneLocalFourier( int me )
{
	MPI_Status status;

	if( DEBUG )	fprintf( stderr, "Node %3d entered DoOneLocalFourier( )\n", me );

	for( int p = 1; p < MAXPERIODS; p++ )
	{
		PPSums[p] = 0.;
	}

	for( int p = 1; p < MAXPERIODS; p++ )
	{
		float omega = F_2_PI/(float)p;;
		for( int t = 0; t < PPSize; t++ )
		{
			float time = (float)(t + me*PPSize);
			PPSums[p] += PPSignal[t] * sinf( omega*time );
		}
	}

}
