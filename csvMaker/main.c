#include <stdio.h>
#include <stdlib.h>
#define japanese 66
#define male 67
#define female 36

int main()
{
    FILE * output;
    int i;

    output = fopen("C:\\opencv\\imagefaces\\people.csv","w");

    for(i=1;i<male;i++){
        fprintf(output,"C:\\opencv\\imagefaces\\male (%d).png;1\n",i);
    }
    for(i=1;i<female;i++){
        fprintf(output,"C:\\opencv\\imagefaces\\female (%d).png;0\n",i);
    }

    fclose(output);

    return 0;
}
