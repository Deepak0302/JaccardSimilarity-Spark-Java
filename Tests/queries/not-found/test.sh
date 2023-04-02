
# run this file from the CS1003-P4 directory
# by running: Tests/queries/not-found/test.sh

export CLASSPATH=${CLASSPATH}:"/cs/studres/CS1003/0-General/spark/*":.

java CS1003P4 Tests/data "to be or not to be" 0.75 > Tests/queries/not-found/stdout 2> Tests/queries/not-found/stderr

LC_ALL=C sort Tests/queries/not-found/stdout -o Tests/queries/not-found/stdout

diff -y Tests/queries/not-found/stdout Tests/queries/not-found/expected.txt
