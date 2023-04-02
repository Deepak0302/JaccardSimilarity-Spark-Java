
# run this file from the CS1003-P4 directory
# by running: Tests/queries/tree/test.sh

export CLASSPATH=${CLASSPATH}:"/cs/studres/CS1003/0-General/spark/*":.

java CS1003P4 Tests/data "hide the christmas tree carefully" 0.75 > Tests/queries/tree/stdout 2> Tests/queries/tree/stderr

LC_ALL=C sort Tests/queries/tree/stdout -o Tests/queries/tree/stdout

diff -y Tests/queries/tree/stdout Tests/queries/tree/expected.txt
