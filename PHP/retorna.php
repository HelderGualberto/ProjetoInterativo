<?php


$input = fopen("asdf.txt","r") or die ("Unable to open file");

$temperature = fread($input,filesize($input)) ;

fclose($input);

echo "-".$temperature."-";

?>