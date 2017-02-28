/* File:     dijkstra.c
 * Purpose:  Implement Dijkstra's algorithm for solving the single-source
 *           shortest path problem:  find the length of the shortest path
 *           between a specified vertex and all other vertices in a
 *           directed graph.
 *
 * Input:    n, the number of vertices in the digraph
 *           mat, the adjacency matrix of the digraph
 * Output:   A list showing the cost of the shortest path
 *           from vertex 0 to every other vertex in the graph.
 *
 * Compile:  gcc -g -Wall -o dijkstra dijkstra.c
 * Run:      ./dijkstra
 *           For large matrices, put the matrix into a file with n as
 *           the first line and run with ./dijkstra < large_matrix
 *
 * Notes:
 * 1.  Edge lengths should be nonnegative.
 * 2.  The distance from v to w may not be the same as the distance from
 *     w to v.
 * 3.  If there is no edge between two vertices, the length is the constant
 *     INFINITY.  So input edge length should be substantially less than
 *     this constant.
 * 4.  The cost of travelling from a vertex to itself is 0.  So the adjacency
 *     matrix has zeroes on the main diagonal.
 * 5.  No error checking is done on the input.
 * 6.  The adjacency matrix is stored as a 1-dimensional array and subscripts
 *     are computed using the formula:  the entry in the ith row and jth
 *     column is mat[i*n + j]
 */
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>

const int INFINITY = 1000000;

MPI_Datatype block(int n, int loc_n);
void Read_matrix(int loc_mat[], int n, int loc_n,
                 MPI_Datatype blk_col_mpi_t, int my_rank, MPI_Comm comm);
void Print_matrix(int loc_mat[], int n, int loc_n,
                  MPI_Datatype blk_col_mpi_t, int my_rank, MPI_Comm comm);
int Global_vertex(int loc_u, int my_rank, int loc_n);
void Dijkstra(int loc_mat[], int loc_dist[],
              int loc_pred[], int n, int loc_n,
              int my_rank, MPI_Comm comm);
int Find_min_dist(int loc_dist[], int loc_known[], int loc_n);
void Print_dists(int loc_dist[], int n, int loc_n, int my_rank, MPI_Comm comm);
void Print_paths(int loc_pred[], int n, int loc_n,
                 int my_rank, MPI_Comm comm);


int main(int argc, char *argv[]) {
    freopen("1000.txt", "r", stdin);
    //freopen("output.txt", "w", stdout);
    
    int  n, p;//p evenly divide n
    int loc_n;
    double a;
    int *loc_mat, *loc_dist, *loc_pred;
    int my_rank;
    MPI_Comm comm;
    MPI_Datatype blk_col_mpi_t;
    MPI_Init(&argc, &argv);
    comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &p);
    MPI_Comm_rank(comm, &my_rank);
    
//  printf("How many vertices?\n");
 //   scanf("%d", &n);
    
    if(my_rank == 0){
        a = MPI_Wtime();
        scanf("%d", &n);
    }
        MPI_Bcast(&n, 1, MPI_INT, 0, comm);
    
    loc_n = n /p;
    loc_mat = malloc(n*loc_n*sizeof(int));
    loc_dist = malloc(loc_n*sizeof(int));
    loc_pred = malloc(loc_n*sizeof(int));
    
    blk_col_mpi_t = block(n, loc_n);

   // printf("Enter the matrix\n");
    Read_matrix(loc_mat, n, loc_n, blk_col_mpi_t, my_rank, comm);
    
    Dijkstra(loc_mat, loc_dist, loc_pred, n, loc_n, my_rank, comm);
    
    //printf("The distance from 0 to each vertex is:\n");
    Print_dists(loc_dist, n, loc_n, my_rank, comm);
    //printf("The shortest path from 0 to each vertex is:\n");
    Print_paths(loc_pred, n, loc_n, my_rank, comm);
    
    if(my_rank == 0){
        double b = MPI_Wtime();//到这结束
        double c = b - a;//算出来的单位是毫秒
        printf("timeUse: %f /n", c);
        //printf("time=%f\n",(double)c)/18.2);
    }
    
    free(loc_mat);
    free(loc_dist);
    free(loc_pred);

    MPI_Type_free(&blk_col_mpi_t);
    
    MPI_Finalize();
    
   return 0;
}  /* main */


MPI_Datatype block(int n, int loc_n) {
    MPI_Aint lb, extent;
    MPI_Datatype block_mpi_t;
    MPI_Datatype first_bc_mpi_t;
    MPI_Datatype blk_col_mpi_t;
    
    MPI_Type_contiguous(loc_n, MPI_INT, &block_mpi_t);
    MPI_Type_get_extent(block_mpi_t, &lb, &extent);
    
    MPI_Type_vector(n, loc_n, n, MPI_INT, &first_bc_mpi_t);
    MPI_Type_create_resized(first_bc_mpi_t, lb, extent,
                            &blk_col_mpi_t);
    MPI_Type_commit(&blk_col_mpi_t);
    
    MPI_Type_free(&block_mpi_t);
    MPI_Type_free(&first_bc_mpi_t);
    
    return blk_col_mpi_t;
}


/*-------------------------------------------------------------------
 * Function:  Read_matrix
 * Purpose:   Read in the adjacency matrix
 * In arg:    n
 * Out arg:   mat
 */
void Read_matrix(int loc_mat[], int n, int loc_n,
                 MPI_Datatype blk_col_mpi_t, int my_rank, MPI_Comm comm) {
    int* mat = NULL, i, j;
    
    if (my_rank == 0) {
        mat = malloc(n*n*sizeof(int));
        for (i = 0; i < n; i++)
            for (j = 0; j < n; j++)
                scanf("%d", &mat[i*n + j]);
    }
    
    MPI_Scatter(mat, 1, blk_col_mpi_t,
                loc_mat, n*loc_n, MPI_INT, 0, comm);
    
    if (my_rank == 0) free(mat);
}   /* Read_matrix */

/*-------------------------------------------------------------------
 * Function:  Print_matrix
 * Purpose:   Print the contents of the matrix
 * In args:   mat, n
 */
void Print_matrix(int loc_mat[], int n, int loc_n,
                  MPI_Datatype blk_col_mpi_t, int my_rank, MPI_Comm comm) {
    int* mat = NULL, i, j;
    
    if (my_rank == 0) mat = malloc(n*n*sizeof(int));
    MPI_Gather(loc_mat, n*loc_n, MPI_INT,
               mat, 1, blk_col_mpi_t, 0, comm);
    if (my_rank == 0) {
        for (i = 0; i < n; i++) {
            for (j = 0; j < n; j++)
                if (mat[i*n + j] == INFINITY)
                    printf(" i ");
                else
                    printf("%2d ", mat[i*n + j]);
            printf("\n");
        }
        free(mat);
    }
} /* Print_matrix */
/*-------------------------------------------------------------------
 * Function:    Global_vertex
 */
int Global_vertex(int loc_u, int my_rank, int loc_n){
    return loc_u + my_rank*loc_n;
}

/*-------------------------------------------------------------------
 * Function:    Dijkstra
 * Purpose:     Apply Dijkstra's algorithm to the matrix mat
 * In args:     n:  the number of vertices
 *              mat:  adjacency matrix for the graph
 * Out args:    dist:  dist[v] = distance 0 to v.
 *              pred:  pred[v] = predecessor of v on a
 *                  shortest path 0->v.
 */
void Dijkstra(int loc_mat[], int loc_dist[],
              int loc_pred[], int n, int loc_n,
              int my_rank, MPI_Comm comm) {
   int i, u, dist_u, new_dist;
   int loc_v, loc_u, loc_known[loc_n];
   /* my_min[0] = loc_dist_u, my_min[1] = Global_vertex(loc_u)*/
   int my_min[2], glbl_min[2];
    double start, end;

   for (loc_v = 0; loc_v < loc_n; loc_v++){
     loc_dist[loc_v] = loc_mat[loc_v];
     loc_pred[loc_v] = 0;
     loc_known[loc_v] = 0;
   }
   /* Initialize the root node */
   if(my_rank == 0) loc_known[0] = 1;


   /* On each pass find an additional vertex */
   /* whose distance to 0 is known           */
   for (i = 1; i < n; i++) {
       loc_u = Find_min_dist(loc_dist, loc_known, loc_n);
       
       if(loc_u == -1){
           my_min[0] = INFINITY; my_min[1] = -1;//didn't find one
       }
       else{
           my_min[0] = loc_dist[loc_u]; my_min[1] = Global_vertex(loc_u, my_rank, loc_n);
       }
       
       if( my_rank == 0){
           start = MPI_Wtime();
           MPI_Allreduce(my_min, glbl_min, 1, MPI_2INT, MPI_MINLOC, comm);
           end = MPI_Wtime();
           printf("timeUse4Allreduce: %f /n", end - start);
       }
       else
           MPI_Allreduce(my_min, glbl_min, 1, MPI_2INT, MPI_MINLOC, comm);
       
       /* finded the min, and make it known */
       dist_u = glbl_min[0];
       u = glbl_min[1];
       
       /* if u is in my matrix, set it as known */
       if(u/loc_n == my_rank) loc_known[loc_u] = 1;

      for (loc_v = 0; loc_v < loc_n; loc_v++)
         if (!loc_known[loc_v]) {
            new_dist = dist_u + loc_mat[u*loc_n + loc_v];
            if (new_dist < loc_dist[loc_v]) {
               loc_dist[loc_v] = new_dist;
               loc_pred[loc_v] = u;
            }
         }
   } /* for i */
}  /* Dijkstra */

/*-------------------------------------------------------------------
 * Function:    Find_min_dist
 * Purpose:     Find the vertex u with minimum distance to 0
 *              (dist[u]) among the vertices whose distance
 *              to 0 is not known.
 * In args:     dist:  dist[v] = current estimate of distance
 *                 0->v
 *              known:  whether the minimum distance 0-> is
 *                 known
 *              n:  the total number of vertices
 * Ret val:     The vertex u whose distance to 0, dist[u]
 *              is a minimum among vertices whose distance
 *              to 0 is not known.
 */
int Find_min_dist(int loc_dist[], int loc_known[], int loc_n) {
   int loc_v, loc_u=-1, best_so_far = INFINITY;

   for (loc_v = 0; loc_v < loc_n; loc_v++)
      if (!loc_known[loc_v])
         if (loc_dist[loc_v] < best_so_far) {
            loc_u = loc_v;
            best_so_far = loc_dist[loc_v];
         }

   return loc_u;
}  /* Find_min_dist */


/*-------------------------------------------------------------------
 * Function:    Print_dists
 * Purpose:     Print the length of the shortest path from 0 to each
 *              vertex
 * In args:     n:  the number of vertices
 *              dist:  distances from 0 to each vertex v:  dist[v]
 *                 is the length of the shortest path 0->v
 */
void Print_dists(int loc_dist[], int n, int loc_n, int my_rank, MPI_Comm comm) {
    int v;
    int *dist = NULL;
    
    if(my_rank == 0)
        dist = malloc(n*sizeof(int));
    MPI_Gather(loc_dist, loc_n, MPI_INT, dist, loc_n, MPI_INT, 0, comm);
    //notice that the reccount is loc_n, rather than n
    
    if(my_rank == 0){
    printf("  v    dist 0->v\n");
    printf("----   ---------\n");

    for (v = 1; v < n; v++)
        printf("%3d       %4d\n", v, dist[v]);
    printf("\n");
    
    free(dist);

    }
} /* Print_dists */


/*-------------------------------------------------------------------
 * Function:    Print_paths
 * Purpose:     Print the shortest path from 0 to each vertex
 * In args:     n:  the number of vertices
 *              pred:  list of predecessors:  pred[v] = u if
 *                 u precedes v on the shortest path 0->v
 */
void Print_paths(int loc_pred[], int n, int loc_n,
                 int my_rank, MPI_Comm comm) {
    int *pred = NULL;
    int v, w, *path = NULL, count, i;
    
    if(my_rank == 0)
        pred =  malloc(n*sizeof(int));
    MPI_Gather(loc_pred, loc_n, MPI_INT, pred, loc_n, MPI_INT, 0, comm);
    
    if(my_rank == 0){
        path = malloc(n*sizeof(int));
        
        printf("  v     Path 0->v\n");
        printf("----    ---------\n");
        
        for (v = 1; v < n; v++) {
            printf("%3d:    ", v);
            count = 0;
            w = v;
            while (w != 0) {
                path[count] = w;
                count++;
                w = pred[w];
            }
            printf("0 ");
            for (i = count-1; i >= 0; i--)
                printf("%d ", path[i]);
            printf("\n");
        }
    }
    
    free(pred);
    free(path);
}  /* Print_paths */
