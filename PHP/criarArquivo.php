<?php

$myfile = fopen("temperature.txt", "w") or die("Unable to open file!");

$temperature = $_GET["temperature"];

fwrite($myfile, $temperature);

echo "Saved sucefully";

fclose($myfile);

?> 
