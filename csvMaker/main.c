#include <stdio.h>
#include <stdlib.h>
#define male 66
#define female 77

int main()
{
    FILE * output;
    int i;

    output = fopen("C:\\opencv\\people.csv","w");

    for(i=1;i<female;i++){
        fprintf(output,"C:\\opencv\\m\\woman (%d).png;0\n",i);
    }
    for(i=1;i<male;i++){
        fprintf(output,"C:\\opencv\\m\\men (%d).png;1\n",i);
    }

    fclose(output);

    return 0;
}
