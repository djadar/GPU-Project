
use strict;

my($line);
my($size);
my($Kernel);
my($tlc);
my($Gflops);

print "Size, Kernel, tlc, Gflops\n" ;
while($line=<>) {
    chomp $line;
    if($line =~/^Size: ([\d\.]*)$/) {
        $size = $1; next;
    } 
    if($line =~/^kernel: ([\d\.]*)$/) {
        $Kernel = $1; next;
    } 
    if($line =~/^Processing time: ([\d\.]*)$/) {
        $Kernel = $1; next;
    } 
    if($line =~/^GFLOPS: ([\d\.]*)$/) {
        $Gflops=$1; 
        print "$size, $Kernel, $tlc, $Gflops\n";
        next;
    }
}

