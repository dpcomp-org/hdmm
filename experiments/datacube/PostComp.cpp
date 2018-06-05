#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <string>
#include <queue>
#include <time.h>
#include <map>
#include <math.h>
#include <cstring>

using namespace std;
#define _CRT_SECURE_NO_WARNINGS

#define MAXD 12
#define MAXL 4096
#define NA -1

#define uint64 unsigned __int64
#define int64 __int64

#ifdef __unix
#define fopen_s(pFile,filename,mode) ((*(pFile))=fopen((filename),  (mode)))==NULL
#endif

const char* file_table = "../table.txt";
const char* file_publish_all = "../publish_all.txt";
const char* file_publish_basecell = "../publish_basecell.txt";
const char* file_publish_cuboid = "../publish_cuboid.txt";
const char* file_query = "../query.txt";

struct CUBOID {
    int id;
    int num;
};

//#define CUBE_STRUCTURE map<int, double>
#define CUBE_STRUCTURE float*
#define CUBE_STRUCTURE_IND int
#define CUBE_STRUCTURE_NODE float

//Output the cubes to the followings files
const char* file_cube_all     = "cube_all.txt";
const char* file_cube_all_con = "cube_all_con.txt";
const char* file_cube_base    = "cube_base.txt";
const char* file_cube_opt     = "cube_opt.txt";
const char* file_cube_con     = "cube_con.txt";
const char* file_cube_opt2    = "cube_opt2.txt";
const char* file_cube_con2    = "cube_con2.txt";

const int _1 = 1;

#define PAR_IN 0

//Input files
char* file_table_dim   = "table_dim_large.txt";
char* file_opt_select  = "../select_large.txt";
char* file_opt_select2 = "../select_large2.txt";
char* file_opt_select3 = "../select_large3.txt";

//Output statistics
char* file_result      = "stat_large.txt";

CUBE_STRUCTURE_NODE eps = 1;

char *par = "01000000";// "11111111";

int n_dim; // number of dimensions
int max_dim = MAXD; // max number of dimensions
int L;
int card[MAXD]; // cardinality of each dimension
int n_cells; // number of cells

int n_select_cuboids; //number of selected cuboids to materialize
int select_cuboids[MAXL]; //selected cuboids
CUBE_STRUCTURE_NODE noise[MAXL]; //selected noise

//int n_base_cells; // number of base cells
//int** base_cell; // base cells

// Kall
CUBE_STRUCTURE data_all;

// Kbase
CUBE_STRUCTURE data_base;

// Kpart
CUBE_STRUCTURE data_opt;

// Consistency
CUBE_STRUCTURE data_con;
int _cell[MAXD], deg[MAXL], ratio[MAXL];

// Record errors in the published cuboids
CUBE_STRUCTURE_NODE var_max;
CUBE_STRUCTURE_NODE var_sum;
int                 var_n;

float max(float xx, float yy)
{
	if (xx<yy)
		return yy;
	else
		return xx;
}

// Calculate the number of dimensions in cuboid query
int querySize(int query)
{
	int ret=0;
	for (int i=0; i<n_dim; i++)
		if (query&(1<<i))
			ret++;
	return ret;
}

void readTableDim()
{
	ifstream fin;
	fin.open(file_table_dim, ios::in);
	if (fin.fail())
	{
		cout << "Table-dimension file does not exist: " << file_table_dim << endl;
		exit(1);
	}

	n_cells = 1;
	fin >> n_dim;
	for (int d = 0; d < n_dim; d++)
	{
		fin >> card[d];
		n_cells *= (card[d]+1);
	}

	L = 0;
	for (int i = 0; i < (_1<<n_dim); i ++)
		if (querySize(i) <= max_dim)
			L ++;
	fin.close();
}

int stringToInt(char* tmp)
{
	int base = 1;
	int res = 0;
	for (unsigned int d = 0; d < strlen(tmp); d++)
	{
		res += base*(tmp[d]=='1');
		base = base*2;
	}
	return res;
}

void readSelectedCuboids(char *filename)
{
	ifstream fin;
	fin.open(filename, ios::in);
	if (fin.fail())
	{
		cout << "Selected-cuboids file does not exist: " << file_opt_select << endl;
		exit(1);
	}
	fin >> n_select_cuboids;
	char* tmp = new char[n_dim+10];
	for (int i = 0; i < n_select_cuboids; i++)
	{
		fin >> tmp;
		select_cuboids[i] = stringToInt(tmp);
		noise[i]=(CUBE_STRUCTURE_NODE)n_select_cuboids;
	}
	fin.close();
}

void readSelectedCuboidsNoise(char *filename)
{
	ifstream fin;
	fin.open(filename, ios::in);
	if (fin.fail())
	{
		cout << "Selected-cuboids file does not exist: " << file_opt_select << endl;
		exit(1);
	}
	fin >> n_select_cuboids;
	char* tmp = new char[n_dim+10];
	for (int i = 0; i < n_select_cuboids; i++)
	{
		fin >> tmp;
		select_cuboids[i] = stringToInt(tmp);
		fin >> noise[i];
	}
	fin.close();
}

bool isSelectCuboid(int cuboid)
{
	for (int i = 0; i < n_select_cuboids; i++)
	{
		if (select_cuboids[i] == cuboid) return true;
	}
	return false;
}

CUBE_STRUCTURE_NODE noiseLaplace(CUBE_STRUCTURE_NODE b)
{
	int temp=rand();
	while (temp==0||temp==RAND_MAX)
		temp=rand();
	CUBE_STRUCTURE_NODE U = ((CUBE_STRUCTURE_NODE)(temp)) / ((CUBE_STRUCTURE_NODE)(RAND_MAX)) - (CUBE_STRUCTURE_NODE)0.5;
	CUBE_STRUCTURE_NODE sign;
	if (U < 0) sign = -1.0; else sign = 1.0;
	return (CUBE_STRUCTURE_NODE)(-1.0*b/eps)*sign*log(1-2*fabs(U));
}

int indexCell(int cell[MAXD])
{
	int ret=0;
	int bas=1;
	for (int i = 0; i < n_dim; i++)
	{
		ret += cell[i]*bas;
		bas *= card[i]+1;
	}
	return ret;
}

void publishAllDfs(int d, int cell[MAXD], CUBE_STRUCTURE data)
{
	if (d == n_dim)
		data[indexCell(cell)]=noiseLaplace((CUBE_STRUCTURE_NODE)L);
	else
		for (int i = 0; i <= card[d]; i++)
		{
			cell[d] = i;
			publishAllDfs(d+1, cell, data);
		}
}

void publishBaseDfs(int d, int cell[MAXD], CUBE_STRUCTURE data)
{
	if (d == n_dim)
		data[indexCell(cell)]=noiseLaplace(1);
	else
		for (int i = 1; i <= card[d]; i++)
		{
			cell[d] = i;
			publishBaseDfs(d+1, cell, data);
		}
}

void publishOptDfs(int d, int cell[MAXD], int cuboid, CUBE_STRUCTURE_NODE noise_cuboid, CUBE_STRUCTURE data)
{
	if (d == n_dim)
		data[indexCell(cell)]=noiseLaplace(noise_cuboid);
	else
	{
		if (cuboid&(_1<<d))
			for (int i = 1; i <= card[d]; i++)
			{
				cell[d] = i;
				publishOptDfs(d+1, cell, cuboid, noise_cuboid, data);
			}
		else
		{
			cell[d] = 0;
			publishOptDfs(d+1, cell, cuboid, noise_cuboid, data);
		}
	}
}

void publishAll(CUBE_STRUCTURE data)
{
	// data.clear();
	memset(data, 0, n_cells*sizeof(CUBE_STRUCTURE_NODE));
	int cell[MAXD];
	srand ( (unsigned int)time(NULL) );
	publishAllDfs(0, cell, data);
}

void publishBase(CUBE_STRUCTURE data)
{
	// data.clear();
	memset(data, 0, n_cells*sizeof(CUBE_STRUCTURE_NODE));
	int cell[MAXD];
	srand ( (unsigned int)time(NULL) );
	publishBaseDfs(0, cell, data);
}

void publishOpt(CUBE_STRUCTURE data)
{
	// data.clear();
	memset(data, 0, n_cells*sizeof(CUBE_STRUCTURE_NODE));
	int cell[MAXD];
	srand ( (unsigned int)time(NULL) );
	for (int i = 0; i < n_select_cuboids; i++)
		publishOptDfs(0, cell, select_cuboids[i], noise[i], data);
}

int useCuboid(int query, int cuboid)
{
	int base = 1;
	int res = 1;
	for (int d = 0; d < n_dim; d++)
	{
		int base = _1<<d;
		int in_query = (query & base);
		int in_cuboid = (cuboid & base);
		if ((in_query) && (!in_cuboid)) return -1; // query is not in this cuboid
		if ((!in_query) && (in_cuboid)) res = res * card[d];
	}
	return res;
}

// For each d-dim cuboid, select an (d+1)-dim cuboid from which it is computed... with the minimal noise magnification
// n_cuboids and cuboids[]: |L_pre| and L_pre
// ns[]: DP noise added into each cuboid in L_pre
void selectCuboid(int use_cuboid[MAXL], int n_cuboids, int cuboids[MAXL], CUBE_STRUCTURE_NODE ns[MAXL])
{
	int* min_cuboid = new int[MAXL];
	CUBE_STRUCTURE_NODE* min_mag = new CUBE_STRUCTURE_NODE[MAXL];

	for (int i=0; i<(_1<<n_dim); i++)
	{
		min_cuboid[i]=NA;
		min_mag[i]=NA;
		use_cuboid[i]=NA;
	}

	for (int i=(_1<<n_dim)-1; i>=0; i--)
		for (int j=0; j<n_cuboids; j++)
			if ((i&cuboids[j])==i)
			{
				CUBE_STRUCTURE_NODE temp=useCuboid(i,cuboids[j])*ns[j];
				if (min_mag[i]==NA||temp<min_mag[i])
				{
					min_mag[i]=temp;
					min_cuboid[i]=cuboids[j];
				}
			}

	for (int i=(_1<<n_dim)-1; i>=0; i--)
	{
		if (min_cuboid[i]==i)
			use_cuboid[i]=i;
		else
			for (int d=0; d<n_dim; d++)
				if ((i&(_1<<d))==0)
					if (min_cuboid[i]==min_cuboid[i|(_1<<d)])
						use_cuboid[i]=(i|(_1<<d));
	}

	delete min_cuboid;
	delete min_mag;
}

// From a d-dim cuboid cuboid, compute (d-1)-dim cuboids
void compCuboid(int cuboid, int cell[MAXD], int i, CUBE_STRUCTURE data, int use_cuboid[MAXL])
{
	if (i==n_dim)
	{
		CUBE_STRUCTURE_IND ind = indexCell(cell);
		for (int j=0; j<n_dim; j++)
			if (cuboid&(_1<<j))
			{
				int pcuboid=(cuboid^(_1<<j));
				if (use_cuboid[pcuboid]==cuboid)
				{
					int temp = cell[j];
					cell[j] = 0;
					CUBE_STRUCTURE_IND ind2 = indexCell(cell);
					cell[j] = temp;

					/*
					if (data.find(ind2)==data.end())
					 	data[ind2] = 0;
					*/
					data[ind2] += data[ind];

					CUBE_STRUCTURE_NODE temp1=data[ind];
					CUBE_STRUCTURE_NODE temp2=data[ind2];
				}
			}
	}
	else
	{
		if ((_1<<i)&cuboid)
		{
			for (cell[i]=1; cell[i]<=card[i]; cell[i]++)
				compCuboid(cuboid, cell, i+1, data, use_cuboid);
		}
		else
		{
			cell[i]=0;
			compCuboid(cuboid, cell, i+1, data, use_cuboid);
		}
	}
}

void compCubeAll(CUBE_STRUCTURE data)
{
	return;
}

void compCubeBase(CUBE_STRUCTURE data)
{
	int n_cuboids;
	int* cuboids = new int[MAXL];
	CUBE_STRUCTURE_NODE* ns =  new CUBE_STRUCTURE_NODE[MAXL];

	int* use_cuboid = new int[MAXL];
	int cell[MAXD];

	n_cuboids=1;
	cuboids[0]=(_1<<n_dim)-1;
	ns[0]=1;
	selectCuboid(use_cuboid, n_cuboids, cuboids, ns);

	for (int i=(_1<<n_dim)-1; i>=0; i--)
		if (use_cuboid[i]!=NA)
			compCuboid(i, cell, 0, data, use_cuboid);
}

void compCubeOpt(CUBE_STRUCTURE data)
{
	int* use_cuboid = new int[MAXL];
	int cell[MAXD];

	selectCuboid(use_cuboid, n_select_cuboids, select_cuboids, noise);

	for (int i=(_1<<n_dim)-1; i>=0; i--)
		if (use_cuboid[i]!=NA)
			compCuboid(i, cell, 0, data, use_cuboid);
}

// Calculate the errors in cuboid
void queryCell(int cuboid, int d, int cell[MAXD], CUBE_STRUCTURE data)
{
	if (d == n_dim)
	{
		/*
		if (data.find(indexCell(cell))==data.end())
		{
			printf("Program Error!");
			exit(1);
		}
		*/
		if (indexCell(cell)>=n_cells)
		{
			printf("Program Error!");
			exit(1);
		}


		CUBE_STRUCTURE_NODE noise=fabs(data[indexCell(cell)]);
		var_sum+=noise;
		var_max=max(var_max,noise);
		var_n++;
	}
	else
	{
		if (cuboid&(_1<<d))
			for (int i = 1; i <= card[d]; i++)
			{
				cell[d] = i;
				queryCell(cuboid, d+1, cell, data);
			}
		else
		{
			cell[d] = 0;
			queryCell(cuboid, d+1, cell, data);
		}
	}
}

// Caculate the errors in cuboid query
void queryCuboid(int query, CUBE_STRUCTURE data)
{
	int cell[MAXD];
	var_max=0;
	var_sum=0;
	var_n=0;
	if (querySize(query)>max_dim)
	{
		var_n = 1;
		return;
	}
	else
		queryCell(query,0,cell,data);
}

void compObs(int i, int cell[MAXD], CUBE_STRUCTURE data, CUBE_STRUCTURE data_des)
{
	if (i == n_dim)
	{
		CUBE_STRUCTURE_NODE cellvalue=0;

		for (int j=0; j<n_select_cuboids; j++)
		{
			for (int k=0; k<n_dim; k++)
				if ((_1<<k)&(select_cuboids[j]))
					_cell[k]=cell[k];
				else
					_cell[k]=0;
			cellvalue+=data[indexCell(_cell)];
		}
		data_des[indexCell(cell)]=cellvalue;
	}
	else
		for (cell[i]=1; cell[i]<=card[i]; cell[i]++)
			compObs(i+1, cell, data, data_des);
}

void compDeg()
{
	for (int i=0; i<(_1<<n_dim); i++)
	{
		int ret=1;
		for (int j=0; j<n_dim; j++)
			if (!(i&(_1<<j)))
				ret*=card[j];
		deg[i]=ret;
	}
}

void compRatio()
{
	for (int cub2=0; cub2<(_1<<n_dim); cub2++)
	{
		int ret=0;
		for (int i=0; i<n_select_cuboids; i++)
		{
			int cub=select_cuboids[i];
			if ((cub2&cub)==cub2)
				ret+=deg[cub|cub2];
		}
		ratio[cub2]=ret;
	}
}

void compConsistentCuboid(int cuboid, int i, int cell[MAXD], CUBE_STRUCTURE data)
{
	if (i==n_dim)
	{
		CUBE_STRUCTURE_NODE aux=0;
		for (int j=0; j<n_select_cuboids; j++)
		{
			int common_ans=select_cuboids[j]&cuboid;
			if (common_ans!=cuboid)
			{
				for (int k=0; k<n_dim; k++)
					if ((_1<<k)&common_ans)
						_cell[k]=cell[k];
					else
						_cell[k]=0;
				aux += data[indexCell(_cell)]*deg[select_cuboids[j]|cuboid];
			}
		}
		CUBE_STRUCTURE_NODE obs = data[indexCell(cell)];
		data[indexCell(cell)]=(obs-aux)/ratio[cuboid];
	}
	else
	{
		if (cuboid&(_1<<i))
		{
			for (cell[i] = 1; cell[i] <= card[i]; cell[i] ++)
				compConsistentCuboid(cuboid, i+1, cell, data);
		}
		else
		{
			cell[i]=0;
			compConsistentCuboid(cuboid, i+1, cell, data);
		}
	}
}

void compConsistentOpt(CUBE_STRUCTURE &data_org, CUBE_STRUCTURE data)
{
	memset(data, 0, n_cells*sizeof(CUBE_STRUCTURE_NODE));
	//data.clear();
	int cell[MAXD];
	compObs(0, cell, data_org, data);
	compCubeBase(data);
	compDeg();
	compRatio();
	for (int i=0; i<(_1<<n_dim); i++)
		compConsistentCuboid(i, 0, cell, data);
}

void compConsistentAll(CUBE_STRUCTURE &data_org, CUBE_STRUCTURE data)
{
	memset(data, 0, n_cells*sizeof(CUBE_STRUCTURE_NODE));
	//data.clear();
	int cell[MAXD];
	n_select_cuboids=0;
	for (int i=0; i<(_1<<n_dim); i++)
		if (querySize(i)<=max_dim)
			select_cuboids[n_select_cuboids++]=i;
	compObs(0, cell, data_org, data);
	compCubeBase(data);
	compDeg();
	compRatio();
	for (int i=0; i<(_1<<n_dim); i++)
		compConsistentCuboid(i, 0, cell, data);
}

void outputCell(int i, int cell[MAXD], CUBE_STRUCTURE data, FILE *fout)
{
	if (i==n_dim)
	{
		for (int j=0; j<n_dim; j++)
			if (cell[j]==0)
				fprintf(fout,"*");
			else
				fprintf(fout,"%c", cell[j]+'A'-1);
		fprintf(fout,": %0.4lf\n", data[indexCell(cell)]);
	}
	else
		for (cell[i]=0; cell[i]<=card[i]; cell[i]++)
			outputCell(i+1, cell, data, fout);
}

void outputCube(CUBE_STRUCTURE data, const char* filename)
{
	FILE *fout;
	fopen_s(&fout, filename, "w");
	int cell[MAXD];
	outputCell(0, cell, data, fout);
	fclose(fout);
}

void copyBase(int d, int cell[MAXD], CUBE_STRUCTURE temp, CUBE_STRUCTURE data)
{
	if (d == n_dim)
		temp[indexCell(cell)] = data[indexCell(cell)];
	else
		for (int i = 1; i <= card[d]; i++)
		{
			cell[d] = i;
			copyBase(d+1, cell, temp, data);
		}
}

int checkConsistency(CUBE_STRUCTURE temp, CUBE_STRUCTURE data)
{
	memset(temp, 0, n_cells*sizeof(CUBE_STRUCTURE_NODE));
	//temp.clear();
	int cell[MAXD];
	copyBase(0, cell, temp, data);
	compCubeBase(temp);

	/*
	int n_cell = 1;
	for (int i=0; i<n_dim; i++)
		n_cell *= (card[i]+1);
	*/
	for (int i=0; i<n_cells; i++)
		if (fabs(temp[i]-data[i])>0.0001)
			return 0;
	return 1;
}

void exeQuery(char *par)
{
	double max0=0,max1=0,max2=0,max3=0,max4=0,max5=0,max6=0,max7=0;
	double sum0=0,sum1=0,sum2=0,sum3=0,sum4=0,sum5=0,sum6=0,sum7=0;
	double tim0=0,tim1=0,tim2=0,tim3=0,tim4=0,tim5=0,tim6=0,tim7=0;
	double cnt=0;
	double m0[MAXD],m1[MAXD],m2[MAXD],m3[MAXD],m4[MAXD],m5[MAXD],m6[MAXD],m7[MAXD];
	double s0[MAXD],s1[MAXD],s2[MAXD],s3[MAXD],s4[MAXD],s5[MAXD],s6[MAXD],s7[MAXD];
	double cn[MAXD];
	for (int i=0; i<MAXD; i++)
	{
		m0[i]=0;
		m1[i]=0;
		m2[i]=0;
		m3[i]=0;
		m4[i]=0;
		m5[i]=0;
		m6[i]=0;
		m7[i]=0;
		s0[i]=0;
		s1[i]=0;
		s2[i]=0;
		s3[i]=0;
		s4[i]=0;
		s5[i]=0;
		s6[i]=0;
		s7[i]=0;
		cn[i]=0;
	}
	for (int query=0; query<(1<<n_dim); query++)
	{
		int sz=querySize(query);
		if (sz<=max_dim)
			cnt++;
		cn[sz]++;
	}

	data_con=new CUBE_STRUCTURE_NODE[n_cells];
	data_all=new CUBE_STRUCTURE_NODE[n_cells];

	if (par[0]=='1'){
	tim0 = clock();
	printf("Running All...");
	publishAll(data_all);
	compCubeAll(data_all);
	printf("Completed!\n");
	tim0 = clock()-tim0;
	printf("Number of Cells: %d...\n", n_cells);
	//outputCube(*data_all, file_cube_all); printf("Completed!\n");
	for (int query=0; query<(1<<n_dim); query++)
	{
		cout << "Evaluating All: cuboid " << query << "/" << (1<<n_dim) << endl;
		int sz=querySize(query);
		queryCuboid(query,data_all);
		max0=max(max0,var_sum/var_n);
		sum0+=var_sum/var_n;
		m0[sz]=max(m0[sz],var_sum/var_n);
		s0[sz]+=var_sum/var_n;
		//max0=max(max0,var_max);
		//sum0+=var_sum;
		//cnt+=var_n;
	}
	}

	if (par[1]=='1'){
	tim1 = clock();
	printf("Enforcing consistency...");
	compConsistentAll(data_all,data_con);
	printf("Completed!\n");
	tim1 = clock()-tim1;
	
	if (checkConsistency(data_all,data_con))
		printf("Consistent!\n");
	else
		printf("Inconsistent!\n");
	
	printf("Number of Cells: %d...\n", n_cells);
	//outputCube(data_con, "test.txt");
	//outputCube(*data_con, file_cube_all_con); printf("Completed!\n");
	for (int query=0; query<(1<<n_dim); query++)
	{
		cout << "Evaluating AllC: cuboid " << query << "/" << (1<<n_dim) << endl;
		int sz=querySize(query);
		queryCuboid(query,data_con);
		max1=max(max1,var_sum/var_n);
		sum1+=var_sum/var_n;
		m1[sz]=max(m1[sz],var_sum/var_n);
		s1[sz]+=var_sum/var_n;
		//max1=max(max1,var_max);
		//sum1+=var_sum;
	}
	}

	delete data_all;

	data_base=new CUBE_STRUCTURE_NODE[n_cells];

	if (par[2]=='1'){
	tim2 = clock();
	printf("Running Base...");
	publishBase(data_base);
	compCubeBase(data_base);
	printf("Completed!\n");
	tim2 = clock()-tim2;
	printf("Number of Cells: %d...\n", n_cells);
	//outputCube(*data_base, file_cube_base); printf("Completed!\n");
	for (int query=0; query<(1<<n_dim); query++)
	{
		cout << "Evaluating Base: cuboid " << query << "/" << (1<<n_dim) << endl;
		int sz=querySize(query);
		queryCuboid(query,data_base);
		max2=max(max2,var_sum/var_n);
		sum2+=var_sum/var_n;
		m2[sz]=max(m2[sz],var_sum/var_n);
		s2[sz]+=var_sum/var_n;
		//max2=max(max2,var_max);
		//sum2+=var_sum;
	}
	}

	delete data_base;

	data_opt=new CUBE_STRUCTURE_NODE[n_cells];

	if (par[3]=='1'){
	tim3 = clock();
	printf("Running BMax...");
	readSelectedCuboids(file_opt_select);
	publishOpt(data_opt);
	compCubeOpt(data_opt);
	printf("Completed!\n");
	tim3 = clock()-tim3;
	printf("Number of Cells: %d...\n", n_cells);
	//outputCube(*data_opt, file_cube_opt); printf("Completed!\n");
	for (int query=0; query<(1<<n_dim); query++)
	{
		cout << "Evaluating BMax: cuboid " << query << "/" << (1<<n_dim) << endl;
		int sz=querySize(query);
		queryCuboid(query,data_opt);
		max3=max(max3,var_sum/var_n);
		sum3+=var_sum/var_n;
		m3[sz]=max(m3[sz],var_sum/var_n);
		s3[sz]+=var_sum/var_n;
		//max3=max(max3,var_max);
		//sum3+=var_sum;
	}
	}

	if (par[4]=='1'){
	tim4 = clock();
	printf("Enforcing consistency...");
	compConsistentOpt(data_opt,data_con);
	printf("Completed!\n");
	tim4 = clock()-tim4;
	
	if (checkConsistency(data_opt,data_con))
		printf("Consistent!\n");
	else
		printf("Inconsistent!\n");
	
	printf("Number of Cells: %d...\n", n_cells);
	//outputCube(*data_con, file_cube_con); printf("Completed!\n");
	for (int query=0; query<(1<<n_dim); query++)
	{
		cout << "Evaluating BMaxC: cuboid " << query << "/" << (1<<n_dim) << endl;
		int sz=querySize(query);
		queryCuboid(query,data_con);
		max4=max(max4,var_sum/var_n);
		sum4+=var_sum/var_n;
		m4[sz]=max(m4[sz],var_sum/var_n);
		s4[sz]+=var_sum/var_n;
		//max4=max(max4,var_max);
		//sum4+=var_sum;
	}
	}

	if (par[5]=='1'){
	tim5 = clock();
	printf("Running PMost...");
	readSelectedCuboids(file_opt_select2);
	publishOpt(data_opt);
	compCubeOpt(data_opt);
	printf("Completed!\n");
	tim5 = clock()-tim5;
	printf("Number of Cells: %d...\n", n_cells);
	//outputCube(*data_opt, file_cube_opt2); printf("Completed!\n");
	for (int query=0; query<(1<<n_dim); query++)
	{
		cout << "Evaluating PMost: cuboid " << query << "/" << (1<<n_dim) << endl;
		int sz=querySize(query);
		queryCuboid(query,data_opt);
		max5=max(max5,var_sum/var_n);
		sum5+=var_sum/var_n;
		m5[sz]=max(m5[sz],var_sum/var_n);
		s5[sz]+=var_sum/var_n;
		//max5=max(max5,var_max);
		//sum5+=var_sum;
	}
	}

	if (par[6]=='1'){
	tim6 = clock();
	printf("Enforcing consistency...");
	compConsistentOpt(data_opt,data_con);
	printf("Completed!\n");
	tim6 = clock()-tim6;
	
	if (checkConsistency(data_opt,data_con))
		printf("Consistent!\n");
	else
		printf("Inconsistent!\n");
	
	printf("Number of Cells: %d...\n", n_cells);
	//outputCube(*data_con, file_cube_con2); printf("Completed!\n");
	for (int query=0; query<(1<<n_dim); query++)
	{
		cout << "Evaluating PMostC: cuboid " << query << "/" << (1<<n_dim) << endl;
		int sz=querySize(query);
		queryCuboid(query,data_con);
		max6=max(max6,var_sum/var_n);
		sum6+=var_sum/var_n;
		m6[sz]=max(m6[sz],var_sum/var_n);
		s6[sz]+=var_sum/var_n;
		//max6=max(max6,var_max);
		//sum6+=var_sum;
	}
	}

	if (par[7]=='1'){
	tim7 = clock();
	printf("Running BMaxG...");
	readSelectedCuboidsNoise(file_opt_select3);
	publishOpt(data_opt);
	compCubeOpt(data_opt);
	printf("Completed!\n");
	tim7 = clock()-tim7;
	printf("Number of Cells: %d...\n", n_cells);
	//outputCube(*data_opt, file_cube_opt2); printf("Completed!\n");
	for (int query=0; query<(1<<n_dim); query++)
	{
		cout << "Evaluating BMaxG: cuboid " << query << "/" << (1<<n_dim) << endl;
		int sz=querySize(query);
		queryCuboid(query,data_opt);
		max7=max(max7,var_sum/var_n);
		sum7+=var_sum/var_n;
		m7[sz]=max(m7[sz],var_sum/var_n);
		s7[sz]+=var_sum/var_n;
		//max7=max(max7,var_max);
		//sum7+=var_sum;
	}
	}

	delete data_opt;
	delete data_con;

	FILE *fout;
	fopen_s(&fout, file_result, "w");
	fprintf(fout,"Algo\t%10s\t%10s\t%10s\t%10s\t%10s\t%10s\t%10s\t%10s\n",
		"All", "AllC", "Base", "BMax", "BMaxC", "PMost", "PMostC", "BMaxG");
	/*
	fprintf(fout,"Time\t%10.4lf\t%10.4lf\t%10.4lf\t%10.4lf\t%10.4lf\t%10.4lf\t%10.4lf\t%10.4lf\n",
		tim0/CLOCKS_PER_SEC+tim2/CLOCKS_PER_SEC, tim1/CLOCKS_PER_SEC, tim2/CLOCKS_PER_SEC+tim2/CLOCKS_PER_SEC, tim3/CLOCKS_PER_SEC+tim2/CLOCKS_PER_SEC,
		tim4/CLOCKS_PER_SEC, tim5/CLOCKS_PER_SEC+tim2/CLOCKS_PER_SEC, tim6/CLOCKS_PER_SEC, tim7/CLOCKS_PER_SEC+tim2/CLOCKS_PER_SEC);
	*/
	fprintf(fout,"Time\t%10.4lf\t%10.4lf\t%10.4lf\t%10.4lf\t%10.4lf\t%10.4lf\t%10.4lf\t%10.4lf\n",
		tim0/CLOCKS_PER_SEC, tim1/CLOCKS_PER_SEC, tim2/CLOCKS_PER_SEC, tim3/CLOCKS_PER_SEC, tim4/CLOCKS_PER_SEC, tim5/CLOCKS_PER_SEC, tim6/CLOCKS_PER_SEC, tim7/CLOCKS_PER_SEC);
	fprintf(fout,"MaxE\t%10.4lf\t%10.4lf\t%10.4lf\t%10.4lf\t%10.4lf\t%10.4lf\t%10.4lf\t%10.4lf\n",
		max0, max1, max2, max3, max4, max5, max6, max7);
	fprintf(fout,"AvgE\t%10.4lf\t%10.4lf\t%10.4lf\t%10.4lf\t%10.4lf\t%10.4lf\t%10.4lf\t%10.4lf\n\n",
		sum0/cnt, sum1/cnt, sum2/cnt, sum3/cnt, sum4/cnt, sum5/cnt, sum6/cnt, sum7/cnt);
	fprintf(fout,"Max Cuboid Error\n");
	for (int i=0; i<=n_dim; i++)
 		fprintf(fout,"%d\t%10.4lf\t%10.4lf\t%10.4lf\t%10.4lf\t%10.4lf\t%10.4lf\t%10.4lf\t%10.4lf\n",
			i, m0[i], m1[i], m2[i], m3[i], m4[i], m5[i], m6[i], m7[i]);
	fprintf(fout,"\n");
	fprintf(fout,"Avg Cuboid Error\n");
	for (int i=0; i<=n_dim; i++)
 		fprintf(fout,"%d\t%10.4lf\t%10.4lf\t%10.4lf\t%10.4lf\t%10.4lf\t%10.4lf\t%10.4lf\t%10.4lf\n",
			i, s0[i]/cn[i], s1[i]/cn[i], s2[i]/cn[i], s3[i]/cn[i], s4[i]/cn[i], s5[i]/cn[i], s6[i]/cn[i], s7[i]/cn[i]);
	fclose(fout);
}

#if PAR_IN==1
int main(int argc, char *argv[])
#else
int main()
#endif
{
#if PAR_IN==1
	file_table_dim   = argv[1];
	file_opt_select  = argv[2];
	file_opt_select2 = argv[3];
	file_opt_select3 = argv[4];
	file_result      = argv[5];
	/*
	sscanf_s(argv[6], "%lf", &eps);
	*/
	sscanf_s(argv[6], "%f",  &eps);
	par              = argv[7];
	if (argc > 8)
		sscanf_s(argv[8], "%d", &max_dim);
	else
		max_dim = MAXD;
#endif
	printf("Reading table structure...");
	readTableDim();
	printf("Completed!\n");
	exeQuery(par);
	return 0;
}
