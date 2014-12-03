<?php

$myFile = fopen("temperatura.txt", "w");

$value = $_GET["ajustar"];

fwrite($myFile,$value);

fclose($myFile);

echo("Saved sucefully");

?>