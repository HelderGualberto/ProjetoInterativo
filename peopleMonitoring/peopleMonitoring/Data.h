#include <iostream>
#include <stdio.h>

class Data{
	
public:
	Data::Data(void);
	void Data::SaveData();

	std::string startTime;//hor�rio de inicializa��o do app;
	std::string endTime;//horario de finaliza��o do app;
	float livedTime;//tempo total de funcionamento do app

	float observedTime; // tempo dentro do intervalo de uma hora em que as pessoas interessadas ficaram olhando a vitrine	
	int numberOfPeople;//Numero total de pessas que passaram em uma hora
	int intrestedPeople;//Numero de pessoas interassadas no intervalo de uma hora

};