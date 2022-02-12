
use strict;

my($line);
my($size);
my($Kernel);
my($choice);
my($Time);
my($Gflops);

print "Size, Kernel, Choice, Time, Gflops\n" ;
while($line=<>) {
    chomp $line;
    if($line =~/^Size: ([\d\.]*)$/) {
        $size = $1; next;
    } 
    if($line =~/^kernel: ([\d\.]*)$/) {
        $Kernel = $1; next;
    }
    if($line =~/^choice: ([\d\.]*)$/) {
        $choice = $1; next;
    } 
    if($line =~/^Time: ([\d\.]*)$/) {
        $Time=$1; next;
    } 
    if($line =~/^	 GFLOPS: ([\d\.]*)$/) {
        $Gflops=$1; 
        print "$size, $Kernel, $choice, $Time, $Gflops\n";
        next;
    }
}
