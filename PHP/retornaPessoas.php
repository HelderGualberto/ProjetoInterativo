<?php

	define("HOST", "172.246.16.27");
	define("USER", "root");
	define("PASS", "8963");
	define("DB", "pi_helder_joao");


class alunosDB extends mysqli {

    public function __construct() {

        parent::__construct(HOST, USER ,PASS , DB);

        if (mysqli_connect_error()) {
            die('Connect Error (' . mysqli_connect_errno() . ') '
                    . mysqli_connect_error());
        }
    }
}


$banco = new alunosDB();
$banco->set_charset("utf8");

    $numberOfPeople = $_REQUEST ['PeopleData'];
	$query = "SELECT * FROM pi_helder_joao.PeopleData"; // banco.tabela
	$st = $banco->query($query);

	while ($row = $st->fetch_object()) {
		
		$result[] = $row;

	}

	echo json_encode($result);
?>