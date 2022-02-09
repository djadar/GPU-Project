
use strict;

my($line);
my($size);
my($Kernel);
my($Time);
my($Gflops);

print "Size, Kernel, Time, Gflops\n" ;
OUTTER: while($line=<>) {
    chomp $line;
    if($line =~/^Size: ([\d\.]*)$/) {
        $size = $1; next;
    } 
    if($line =~/^kernel: ([\d\.]*)$/) {
        $Kernel = $1; next;
    } 
    if($line =~/^(.*)Time: ([\d\.]*)$/) {
        $Time=$1; next OUTTER;
    } 
    if($line =~/^	 GFLOPS: ([\d\.]*)$/) {
        $Gflops=$1; 
        print "$size, $Kernel, $Time, $Gflops\n";
        next;
    }
}

