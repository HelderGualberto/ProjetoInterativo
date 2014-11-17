<?php

	define("HOST", "172.246.16.27");
	define("USER", "root");
	define("PASS", "8963");
	define("DB", "test");


class alunosDB extends mysqli {

    public function __construct() {

        parent::__construct(HOST, USER ,PASS , DB);

        if (mysqli_connect_error()) {
            die('Connect Error (' . mysqli_connect_errno() . ') '
                    . mysqli_connect_error());
        }
    }
}

$latitude = $_REQUEST['latitude'];
$longitude = $_REQUEST['longitude'];
$nome = $_REQUEST['nome'];
$id = $_REQUEST['id'];


$banco = new alunosDB();
$banco->set_charset("utf8");

    $id_aluno = $_REQUEST ['id_aluno'];
	$query = "INSERT INTO joao(id , nome , latitude , longitude ) VALUES ('$id', '$nome' , '$latitude' , '$longitude')";
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