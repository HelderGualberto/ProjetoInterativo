#include <windows.h>
#include <mysql.h>
#include <iostream>
#include <fstream>
#include <sstream>

int main(int argc, char **argv)
{
  MYSQL *con = mysql_init(NULL);

  FILE * input = fopen("C:\\Users\\Helder\\Documents\\GitHub\\ProjetoInterativo\\peopleMonitoring\\appData.txt","r");

  int numberOfPeople,intrestedPeople;
  float time;
  char startTime[20],endTime[20];

  fscanf(input,"%s",startTime);
  getc(input);
  fscanf(input,"%d",&numberOfPeople);
  fscanf(input,"%d",&intrestedPeople);
  fscanf(input,"%f",&time);
  fscanf(input,"%s",endTime);
  getc(input);

  fclose(input);

  printf("Intrested: %d\n",intrestedPeople);
  printf("total: %d\n",numberOfPeople);
  printf("time: %f\n",time);
  printf("start: %s\n",startTime);
  printf("end: %s\n",endTime);

  if (con == NULL)
  {
    fprintf(stderr, "%s\n", mysql_error(con));
    exit(1);
  }
//172.246.16.27
//root
//8963
  if (mysql_real_connect(con, "172.246.16.27", "root", "8963",
    "pi_helder_joao", 0, NULL, 0) == NULL)
  {
    fprintf(stderr, "%s\n", mysql_error(con));
    mysql_close(con);
    exit(1);
  }

  if (mysql_query(con, "INSERT INTO total_pessoas (numero_pessoas) VALUES('666')"))
  {
      fprintf(stderr, "%s\n", mysql_error(con));
      mysql_close(con);
      exit(1);
  }


  mysql_close(con);
  exit(0);
}
