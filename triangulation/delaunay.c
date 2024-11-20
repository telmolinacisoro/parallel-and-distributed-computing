#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define pixel(i, j, w)  (((j)*(w)) +(i))
#define TRUE 1
#define FALSE 0

/* A point in 2D space */
struct Point {
    double x;
    double y;
    double value;
};

/* A triangle defined by three points */
struct Triangle {
    struct Point p1;
    struct Point p2;
    struct Point p3;
};

/* Helper function to output the triangles in the Delaunay Triangulation */
void print_triangles(struct Triangle * triangles, int num_triangles) {
    for (int i = 0; i < num_triangles; i++) {
        printf("(%lf, %lf) (%lf, %lf) (%lf, %lf)\n", 
            triangles[i].p1.x, triangles[i].p1.y,
            triangles[i].p2.x, triangles[i].p2.y,
            triangles[i].p3.x, triangles[i].p3.y);   
    }
}

/* Helper function to calculate the Euclidean distance between two points */
double distance(struct Point * p1, struct Point * p2) {
    double dx = (*p1).x - (*p2).x;
    double dy = (*p1).y - (*p2).y;
    return sqrt(dx*dx + dy*dy);
}

/* Helper function to check if a triangle is clockwise */
int is_ccw(struct Triangle * t) {
    double ax = (*t).p2.x - (*t).p1.x;
    double ay = (*t).p2.y - (*t).p1.y;
    double bx = (*t).p3.x - (*t).p1.x;
    double by = (*t).p3.y - (*t).p1.y;

    double area = ax * by - ay * bx;
    return area > 0;
}

/* Helper function to check if a point is inside a circle defined by three points */
int inside_circle(struct Point * p, struct Triangle * t) {
//      | ax-dx, ay-dy, (ax-dx)² + (ay-dy)² |
//det = | bx-dx, by-dy, (bx-dx)² + (by-dy)² |
//      | cx-dx, cy-dy, (cx-dx)² + (cy-dy)² |

    int clockwise = is_ccw(t);
    
    double ax = (*t).p1.x - (*p).x;
    double ay = (*t).p1.y - (*p).y;
    double bx = (*t).p2.x - (*p).x;
    double by = (*t).p2.y - (*p).y;
    double cx = (*t).p3.x - (*p).x;
    double cy = (*t).p3.y - (*p).y;

    double det = ax*by + bx*cy + cx*ay - ay*bx - by*cx - cy*ax;
    det = (ax*ax + ay*ay) * (bx*cy-cx*by) -
            (bx*bx + by*by) * (ax*cy-cx*ay) +
            (cx*cx + cy*cy) * (ax*by-bx*ay);
    
    if (clockwise)
        return det > 0;
    return det<0;
}

//* Helper function to compute barycentric coordintaes of a point respect a triangle */
void barycentric_coordinates(struct Triangle * t, struct Point * p, double * alpha, double * beta, double * gamma) {
    // Compute the barycentric coordinates of the point with respect to the triangle
    (*alpha) = (((*t).p2.y - (*t).p3.y) * ((*p).x - (*t).p3.x) + ((*t).p3.x - (*t).p2.x) * ((*p).y - (*t).p3.y)) /
                  (((*t).p2.y - (*t).p3.y) * ((*t).p1.x - (*t).p3.x) + ((*t).p3.x - (*t).p2.x) * ((*t).p1.y - (*t).p3.y));
    (*beta) = (((*t).p3.y - (*t).p1.y) * ((*p).x - (*t).p3.x) + ((*t).p1.x - (*t).p3.x) * ((*p).y - (*t).p3.y)) /
                 (((*t).p2.y - (*t).p3.y) * ((*t).p1.x - (*t).p3.x) + ((*t).p3.x - (*t).p2.x) * ((*t).p1.y - (*t).p3.y));
    (*alpha) =(*alpha) > 0 ? (*alpha) : 0;
    (*alpha) =(*alpha) < 1 ? (*alpha) : 1;
    (*beta) = (*beta) > 0 ? (*beta) : 0;
    (*beta) = (*beta) < 1 ? (*beta) : 1;
    (*gamma) = 1.0 - (*alpha) - (*beta);
    (*gamma) = (*gamma) > 0 ? (*gamma) : 0;
    (*gamma) = (*gamma) < 1 ? (*gamma) : 1;
}


/* Helper function to check if a point is inside a triangle (IT CAN BE REMOVED)*/
int inside_triangle(struct Triangle * t, struct Point * p) {
    double alpha, beta, gamma;
    barycentric_coordinates(t, p, &alpha, &beta, &gamma); 
    // Check if the barycentric coordinates are positive and add up to 1
    if (alpha > 0 && beta > 0 && gamma > 0) {
        return 1;
    } else {
        return 0;
    }
}

/* Helper function to save an image */   
void save_image(char *filename, int width, int height, double *image){
   FILE *fp = NULL;
   fp = fopen(filename,"w");
   
   for(int j=0; j<height; ++j){
      for(int i=0; i<width; ++i){
         fprintf(fp,"%f ", image[pixel(i,j,width)]);      
      }
      fprintf(fp,"\n");
   }
   fclose(fp);

}

/* helper function to initialize the points */
void init_points(struct Point* points, int num_points, int width, int height) {
    
    for(int i = 0; i < num_points; i++) {
        points[i].x =  ((double) rand() / RAND_MAX)*width;
        points[i].y =  ((double) rand() / RAND_MAX)*height;
        points[i].value = 0;//(rand() % 10000) / 100.;
        //printf("Point %d [%f,%f]=%f\n", i, points[i].x, points[i].y, points[i].value);
    }
}

/*Function to count how many points you have close (dist<100)*/
void count_close_points(struct Point* points, int num_points) {
    
    //Iterate every possible pair of nodes
    #pragma acc parallel loop copyin(num_points) copy(points[0:num_points])
    for (int i = 0; i < num_points; i++) {
        #pragma acc loop
        for (int j = 0; j < num_points; j++) {
            
            //Avoid checking a node with itself
            if (i != j) {

                //If the distance between the two points is smaller than 100 add 1 to the value of the point
                if ( distance(&points[i], &points[j])< 100) {
                    #pragma acc atomic
                    points[i].value++;
                }
            }
        }
    }
}

/* Function to calculate the Delaunay Triangulation of a set of points */
void delaunay_triangulation(struct Point* points, int num_points, struct Triangle* triangles, int* num_triangles) {
    int max_triangles=num_points*(num_points-1)*(num_points-2)/6;
    *num_triangles=0;
    // Iterate over every possible triangle defined by three points avoiding repetitions
    #pragma acc parallel loop copy(num_triangles[:1]) copyin(points[:num_points]) copyout(triangles[:max_triangles])
    for (int i = 0; i < num_points; i++) {
        #pragma acc loop
        for (int j = i + 1; j < num_points; j++) {
            #pragma acc loop
            for (int k = j + 1; k < num_points; k++) {
                struct Point* p1 = &points[i];
                struct Point* p2 = &points[j];
                struct Point* p3 = &points[k];
                
                // Check if no other of the set is inside the circle circumscribed by the point
                int inside = TRUE;

                   
                for (int m = 0; m < num_points; m++) {
                    if (m != i && m != j && m != k) {
                        struct Point* p = &points[m];
                        if (inside_circle(p, &(struct Triangle){*p1,*p2,*p3})){
                            inside = FALSE;

                        }
                    }
                }
                
                // Add the triangle to the list
                if (inside) {
                    int aux;
                    #pragma acc atomic capture
                    {
                        aux= *num_triangles;
                        (*num_triangles)++;

                    }
                    triangles[aux] = (struct Triangle){*p1, *p2, *p3};
                }
            }
                
        }
    
    }
}


/* Function to store an image of int's between 0 and 100, where points store -1, and empty areas -2, and points inside triangle the average value */
void save_triangulation_image(struct Point* points, int num_points, struct Triangle* triangles, int num_triangles, int width, int height) {
    double* image = (double*)malloc(sizeof(double) * width * height); // Allocate memory for the image
    //Do a copy of the image 
    // Initialize the image with default values
    #pragma acc parallel loop
    for (int i = 0; i < width * height; i++) {
        image[i] = -1;
    }

    #pragma acc parallel loop collapse(2) copy(image[:width*height]) copyin(triangles[:num_triangles],points[:num_points])
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            //pixel position in the 1D array
            int pixel_index = pixel(i, j, width);
            
            // Check if the pixel is already marked as a point (-1)
            if (image[pixel_index]==-1){
                for (int k = 0; k < num_triangles; k++) { 
                    struct Triangle* triangle = &triangles[k];

                    // Check if the pixel is inside the triangle 
                    if (inside_triangle(triangle, &(struct Point) {(double)i, (double)j})) {

                        // Compute the interpolated value based on the triangle's points
                        double alpha, beta, gamma;
                        barycentric_coordinates (triangle, &(struct Point){(double)i, (double)j}, &alpha, &beta, &gamma);
                        double interpolated_value = alpha * triangle->p1.value + beta * triangle->p2.value + gamma * triangle->p3.value; 
                        image[pixel_index] = interpolated_value; 
                        break;

                    }
                }

                for(int k = 0; k <num_points; k++){
                    struct Point* sensor = &points[k];
                    if (abs(sensor->x - i) <= 2 && abs(sensor->y - j) <= 2){
                        image[pixel_index]=101;
                        break;
                    }
                }
            }
        }
    }

    save_image("image.txt", width, height, image);
    
    //free memory
    free(image);
}


int delaunay(int num_points, int width, int height) {
    double start, end;
    int max_num_triangles = num_points*30;
    struct Point * points = (struct Point *) malloc(sizeof(struct Point)*num_points);
    struct Triangle * triangles = (struct Triangle *) malloc(sizeof(struct Triangle)*max_num_triangles);
    printf("Maximum allowed number of triangles = %d\n", num_points*30);
    
    init_points(points, num_points, width, height);

    start = omp_get_wtime();
    count_close_points(points, num_points);
    end = omp_get_wtime();
    printf("Counting close points: %f\n", end-start);

    int num_triangles = 0;
    start = omp_get_wtime();
    delaunay_triangulation(points, num_points, triangles, &num_triangles);
    end = omp_get_wtime();
    printf("Delanuay triangulation: %f\n", end-start);
    
    printf("Number of generated triangles = %d\n", num_triangles);
    //print_triangles(triangles, num_triangles);
    
    start = omp_get_wtime();
    save_triangulation_image(points, num_points, triangles, num_triangles, width, height);
    end = omp_get_wtime();
    printf("Generate image: %f\n", end-start);
    
    //Free memory
    free(points);
    free(triangles);
    return 0;
}
