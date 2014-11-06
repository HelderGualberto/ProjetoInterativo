#include <iostream>
#include <fstream>
#include <sstream>
#include <stdlib.h>

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
	
	//fprintf(output,"%d",livedTime);
	ofstream file;
	file.open("C:\\Users\\Helder\\Documents\\GitHub\\ProjetoInterativo\\peopleMonitoring\\livedTime.txt",ios::out);
	file << livedTime;
	file.close();
	
	FILE * output;
	output = fopen("C:\\Users\\Helder\\Documents\\GitHub\\ProjetoInterativo\\peopleMonitoring\\appData.txt","a");
	fprintf(output,"%s\n",startTime.c_str());
	fprintf(output,"%d\n",numberOfPeople);
	fprintf(output,"%d\n",intrestedPeople);
	fprintf(output,"%f\n",observedTime);
	fprintf(output,"%s\n",endTime.c_str());

	fclose(output);
}