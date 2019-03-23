#include<stdio.h>
#include<string.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>
#include<iostream>
#include<string>
#include<sstream>
#include<vector>
#include<queue>
#include<set>
#include<map>
#include<algorithm>

#define PAR_IN 0
#define INF (0x7FFFFFF)
#define NA (-1)

#define uint64 unsigned __int64
#define int64 __int64

#ifdef __unix
#define fopen_s(pFile,filename,mode) ((*(pFile))=fopen((filename),  (mode)))==NULL
#endif

const int _1=1;

#define MAXD 12
#define MAXL 4096
using namespace std;

// Number of dimensions and their cardinalities
int d, s[MAXD], n_cells;
// Number of cuboids to be output, and their flags (w[i] = 1 if i is to be output)
int N, w[MAXL];

// Magnification ratio
int magg[MAXL][MAXL];

// The set of cuboids in L_pre and the amounts of noise to be added in each of them
int nret, ret[MAXL];
double noise[MAXL];

// The coverage of each cuboid, and the coverage cost cst
int ncv[MAXL], cov[MAXL][MAXL];
double cst[MAXL][MAXL];
int nbl[MAXL], blg[MAXL][MAXL];

// Two flags
int sf[MAXL];
int cf[MAXL];

// char *forbid_cb = "../table_dim_large.txt_fcb.txt";
char *table_dim = "../table_dim_large.txt";
char *output1   = "../select_large.txt";
char *output2   = "../select_large2.txt";
char *output3   = "../select_large3.txt";
char *outputs   = "../theoretical_large.txt";
int ratio=5;
int max_dim;
double eps=1;
FILE *fin, *fout;

// The magnification ratio of computing cuboid j from cuboid i
int mag(int i, int j)
{
	if ((i|j)!=i)
		return INF;

	int ret=1;

	i=i^j;
	for (int k=0; k<d; k++)
	{
		if (i&1)
			ret*=s[k];
		i>>=1;
	}
	return ret;
}

// Calculate the number of dimensions in cuboid query
int querySize(int query)
{
	int ret=0;
	for (int i=0; i<d; i++)
		if (query&(1<<i))
			ret++;
	return ret;
}

// Input
void init_cube_structure()
{
	fopen_s(&fin, table_dim, "r");
	fscanf(fin, "%d", &d);
	N = (_1<<d);
	n_cells = 1;
	for (int i = 0; i < d; i++)
	{
		fscanf(fin, "%d", s+i);
		n_cells *= (s[i]+1);
	}
	max_dim = d;
	for (int i = 0; i < N; i++)
		if (querySize(i) <= max_dim)
			w[i] = 1;
		else
			w[i] = 0;
	fclose(fin);

	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
			magg[i][j] = mag(i, j);

	/*
	int fN, fc;
	fopen_s(&fin, forbid_cb, "r");
	fscanf_s(fin, "%d", &fN);
	for (int i = 0; i < fN; i++);
	{
		fscanf_s(fin, "%d", &fc);
		w[fc] = 0;
	}
	fclose(fin);
	*/
}

// Compute the coverage of each cuboid given theta and |L_pre| = p
void comp_coverage(int theta, int p)
{
	memset(ncv,0,sizeof(int)*MAXL);
	memset(nbl,0,sizeof(int)*MAXL);
	for (int i=N-1; i>=0; i--)
		for (int j=i; j>=0; j--)
			if (w[j]>0)
				if ((i|j)==i)
					if (magg[i][j]*p*p<=theta)
					{
						cov[i][ncv[i]]=j;
						ncv[i]++;
						blg[j][nbl[j]]=i;
						nbl[j]++;
					}
}

// Sort cuboids according to the magnification ratios of compute them from cuboid
void sort_cuboid(int cuboid, int nc, int c[MAXL], double cost[MAXL])
{
	for (int i=0; i<nc-1; i++)
		for (int j=i+1; j<nc; j++)
			if (magg[cuboid][c[i]]>magg[cuboid][c[j]])
				swap(c[i], c[j]);
	for (int i=0; i<nc; i++)
		cost[i]=sqrt((double)magg[cuboid][c[i]]*2);
}

// Computing the coverage of cuboids for the general framework
void comp_coverage_general()
{
	for (int i=0; i<N; i++)
	{
		ncv[i]=0;
		for (int j=0; j<=i; j++)
			if (w[j]>0)
				if ((i|j)==i)
					cov[i][ncv[i]++]=j;
		sort_cuboid(i, ncv[i], cov[i], cst[i]);
	}
}

// Greedy for BoundingMax_General
double greedy_general(int &nret, int ret[MAXL], double noise[MAXL])
{
	int tot=0;
	double tot_cost=0;

	memset(cf,0,sizeof(int)*MAXL);
	memset(sf,0,sizeof(int)*MAXL);

	nret=0;
	double best;
	int selt;
	int selt_noise;
	do
	{
		best=0;
		for (int i=0; i<N; i++)
			if (sf[i]==0)
			{
				int temp=0;
				for (int j=0; j<ncv[i]; j++)
				{
					if (cf[cov[i][j]]==0)
						temp++;
					if (temp/cst[i][j]>best)
					{
						best=temp/cst[i][j];
						selt=i;
						selt_noise=j;
					}
				}
			}
		if (best>0)
		{
			tot_cost+=cst[selt][selt_noise];
			sf[selt]=1;
			for (int j=0; j<=selt_noise; j++)
				if (cf[cov[selt][j]]==0)
				{
					cf[cov[selt][j]]=1;
					tot++;
				}
			ret[nret]=selt;
			noise[nret]=cst[selt][selt_noise];
			nret++;
		}
	}
	while (best>0);
	return tot_cost;
}

// BoundingMax_General: different cuboids with different amounts of noise
double boundingmax_general(int &nret, int ret[MAXL], double noise[MAXL])
{
	comp_coverage_general();
	double theta = greedy_general(nret, ret, noise);
	double privacy = 0;
	for (int i = 0; i < nret; i++)
	{
		noise[i] = theta/noise[i];
		privacy += 1/noise[i];
	}
	// printf("%0.4lf\n", privacy);
	return theta * theta;
}

// Greedy for BoundingMax
int greedy(int ret[MAXL])
{
	for (int i=0; i<N; i++)
		sf[i]=ncv[i];
	memset(cf,0,sizeof(int)*MAXL);

	int nret=0;
	int best,selt;
	do
	{
		best=0;
		for (int i=0; i<N; i++)
			if (sf[i]>best)
			{
				best=sf[i];
				selt=i;
			}
		if (best>0)
		{
			for (int j=0; j<ncv[selt]; j++)
			{
				int k=cov[selt][j];
				if (cf[k]==0)
				{
					cf[k]=1;
					for (int l=0; l<nbl[k]; l++)
						sf[blg[k][l]]--;
				}
			}
			ret[nret]=selt;
			nret++;
		}
	}
	while(best);
	return nret;
}

// BoundingMax
int boundingmax(int &nret,int ret[MAXL])
{
	int thetaL=0, thetaR=N*N, theta, p;

	while (thetaR-thetaL>1)
	{
		theta=(thetaR+thetaL)/2;
		for (p=1; p<=N; p++)
			if (theta>=(p*p))
			{
				comp_coverage(theta, p);
				if (greedy(ret)<=p)
					break;
			}
		if (p<=N)
			thetaR=theta;
		else
			thetaL=theta;
	}
	theta=thetaR;
	for (p=1; p<=N; p++)
		if (theta>=(p*p))
		{
			comp_coverage(theta, p);
			nret=greedy(ret);
			if (nret<=p)
				return theta;
		}
	return -1;
}

// Greedy for PublishingMax
int greedy_p(int ret[MAXL], int p)
{
	for (int i=0; i<N; i++)
		sf[i]=ncv[i];
	memset(cf,0,sizeof(int)*MAXL);

	int sum=0;
	int best,selt;
	for (int ii=0; ii<p; ii++)
	{
		best=0;
		for (int i=0; i<N; i++)
			if (sf[i]>best)
			{
				best=sf[i];
				selt=i;
			}
		if (best>0)
		{
			for (int j=0; j<ncv[selt]; j++)
			{
				int k=cov[selt][j];
				if (cf[k]==0)
				{
					cf[k]=1;
					for (int l=0; l<nbl[k]; l++)
						sf[blg[k][l]]--;
				}
			}
			sum+=best;
			ret[ii]=selt;
		}
	}
	return sum;
}

// PublishingMax
int publishingmax(int theta,int &nret,int ret[MAXL])
{
	int p;
	int val=0;

	for (p=1; p<=N; p++)
		if (theta>=(p*p))
		{
			comp_coverage(theta, p);
			int temp=greedy_p(ret,p);
			if (temp>val)
			{
				val=temp;
				nret=p;
			}
		}
	comp_coverage(theta, nret);
	return greedy_p(ret, nret);
}

// Computing max error, the objective function of Problem 1
int comperr(int nret, int ret[MAXL])
{
	int minerr=nret*nret*2;
	int ans=0;

	for (int j=0; j<N; j++)
		if (w[j] > 0)
		{
			int temp=-1;
			for (int k=0; k<nret; k++)
			{
				int i=ret[k];
				if ((i|j)==i)
				{
					if (temp==-1)
						temp=(int)magg[i][j]*minerr;
					else
						temp=min(temp,(int)magg[i][j]*minerr);
				}
			}
			if (temp==-1)
				return -1;
			ans=max(ans,temp);
		}
	return ans;
}

// Output L_pre
void output(int nret, int ret[MAXL], char *filename)
{
	fopen_s(&fout,filename,"w");
	fprintf(fout,"%d\n",nret);
	for (int i=0; i<nret; i++)
	{
		if (ret[i]==-1)
			fprintf(fout,"-1\n");
		else
		{
			for (int j=0; j<d; j++)
				if (ret[i]&(_1<<j))
					fprintf(fout,"1");
				else
					fprintf(fout,"0");
			fprintf(fout,"\n");
		}
	}
	fclose(fout);
}

// Ouptput L_pre for the generalized algorithm
void output_general(int nret, int ret[MAXL], double noise[MAXL], char *filename)
{
	fopen_s(&fout,filename,"w");
	fprintf(fout,"%d\n",nret);
	for (int i=0; i<nret; i++)
	{
		if (ret[i]==-1)
			fprintf(fout,"-1\n");
		else
		{
			for (int j=0; j<d; j++)
				if (ret[i]&(_1<<j))
					fprintf(fout,"1");
				else
					fprintf(fout,"0");
			fprintf(fout," %0.6lf\n", noise[i]);
		}
	}
	fclose(fout);
}


#if PAR_IN==1
int main(int argc, char *argv[])
#else
int main()
#endif
{
#if PAR_IN==1
	table_dim = argv[1];
	output1   = argv[2];
	output2   = argv[3];
	output3   = argv[4];
	outputs   = argv[5];
	sscanf_s(argv[6], "%d", &ratio);
#endif
	/*
	double *aa = new double[60466176];
	for (int i = 0; i < 60466176; i++)
		aa[i] = i;
	printf("%d\n%d\n", sizeof(float), sizeof(double));
	*/

	init_cube_structure();

	int base1=N*N*2;
	
	int base2=2;
	for (int i=0; i<d; i++)
		base2*=s[i];

	double time_boundingmax = clock();
	double minvar = boundingmax(nret,ret)*2;
	time_boundingmax = clock() - time_boundingmax;
	output(nret,ret,output1);

	double time_publishingmax = clock();
	int maxpub = publishingmax((int)(minvar/2)*ratio/10,nret,ret);
	time_publishingmax = clock() - time_publishingmax;
	double maxpubmaxvar = comperr(nret,ret);
	output(nret,ret,output2);
	
	double time_boundingmax_g = clock();
	double minvar_general = boundingmax_general(nret,ret,noise);
	time_boundingmax_g = clock() - time_boundingmax_g;
	output_general(nret, ret, noise, output3);


	fopen_s(&fout,outputs,"w");
	fprintf(fout,"Number of Cells    : %d\n\n",     n_cells);

	fprintf(fout,"Kall  Max Error    : %0.4lf\n\n", base1/(eps*eps));
	fprintf(fout,"Kbase Max Error    : %0.4lf\n\n", base2/(eps*eps));
	
	fprintf(fout,"BoundMax Time      : %0.4lf\n",   time_boundingmax/CLOCKS_PER_SEC);
	fprintf(fout,"Kpart Max Error    : %0.4lf\n\n", minvar/(eps*eps));

	fprintf(fout,"PublishMost Time   : %0.4lf\n",   time_publishingmax/CLOCKS_PER_SEC);
	fprintf(fout,"If Ratio =         : %0.4lf\n",   (double)ratio/10);
	fprintf(fout,"# of Precise       : %d\n",       maxpub);
	fprintf(fout,"Kpart Max Error    : %0.4lf\n\n", maxpubmaxvar/(eps*eps));

	fprintf(fout,"BoundMaxG Time     : %0.4lf\n",   time_boundingmax_g/CLOCKS_PER_SEC);
	fprintf(fout,"KpartG Max Error   : %0.4lf\n",   minvar_general/(eps*eps));
	fclose(fout);

	return 0;
}
