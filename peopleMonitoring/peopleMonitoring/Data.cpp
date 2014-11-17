#include <iostream>
#include <fstream>
#include <sstream>
#include <stdlib.h>
#include <mysql.h>

//Vieira's Database
//172.246.16.27
//root
//8963

using namespace std;

class Data{
public :
	Data(void);
	void Data::SaveData();

	string startTime;//horário de inicialização do app;
	string endTime;//horario de finalização do app;
	float livedTime;//tempo total de funcionamento do app
	float observedTime; // tempo dentro do intervalo de uma hora em que as pessoas interessadas ficaram olhando a vitrine	
	int numberOfPeople;//Numero total de pessas que passaram em uma hora
	int intrestedPeople;//Numero de pessoas interassadas no intervalo de uma hora
	int Women,Men;// Numero de mulhers e homens que se interessaram.
};

Data::Data(void){
	FILE* input;
	input = fopen("C:\\Users\\Helder\\Documents\\GitHub\\ProjetoInterativo\\peopleMonitoring\\livedTime.txt","r");
	int value;
	fscanf(input,"%d",&value);

	livedTime = (float)value;
	//printf("lived time: %f",livedTime);
	fclose(input);
}

void Data::SaveData(){


	ofstream file;
	file.open("C:\\Users\\Helder\\Documents\\GitHub\\ProjetoInterativo\\peopleMonitoring\\livedTime.txt",ios::out);
	file << livedTime;
	file.close();

	MYSQL *con = mysql_init(NULL);

	  if (con == NULL)
	  {
		fprintf(stderr, "%s\n", mysql_error(con));
		exit(1);
	  }
	
	  if (mysql_real_connect(con, "172.246.16.27", "root", "8963",
		"pi_helder_joao", 0, NULL, 0) == NULL)
	  {
		fprintf(stderr, "%s\n", mysql_error(con));
		mysql_close(con);
		exit(1);
	  }

	  std::ostringstream query;

	  query << "INSERT INTO PeopleData (TotalPeople,IntrestedPeople,Woman,Men,ObservedTime,StartTime,EndTime) VALUES('"	
		  << numberOfPeople << "','" 
		  << intrestedPeople << "','" 
		  << Women << "','" 
		  << Men <<"','"
		  << observedTime << "','" 
		  << startTime << "','" 
		  << endTime <<"')";

	 // cout <<  query.str().c_str() << endl;

	  if(mysql_query(con, query.str().c_str()))
	  {
		  fprintf(stderr, "%s\n", mysql_error(con));
		  mysql_close(con);
		  exit(1);
	  }


	  mysql_close(con);

}