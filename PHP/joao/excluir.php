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


	$query = "DELETE FROM pi_helder_joao.PeopleData";


	$st = $banco->query($query);

	if ($banco->affected_rows > 0) {
		
		$result = 1;
	}
	else
	{
		$result = 0;
	}

	echo json_encode($result);

?>