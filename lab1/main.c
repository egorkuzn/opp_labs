#include <stdio.h>
#include <stdlib.h>

void free_func(int* a){
    free(a);
}

int main(){
    int* a = (int*)malloc(5 * sizeof(int));
    a[2] = 5;
    printf("%d\n", a[2]);
    free_func(a);
    printf("%d\n", a[2]);
    return 0;
}
