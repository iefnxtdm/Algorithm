#--Makefile for bcloud--
#the real makefile is
#/home/sunstrike/bc_out/bc_cache/goodcoder_sunminqi_C++-id-15633/bcloud.makefile_main
#/home/sunstrike/bc_out/bc_cache/goodcoder_sunminqi_C++-id-15633/bcloud.makefile_deps
all:
	bcloud local --workspace=reuse --build=all -j2 /home/sunstrike/goodcoder/sunminqi/C++-id-15633 --username=sunminqi

deps:
	bcloud local --workspace=reuse --build=deps -j2 /home/sunstrike/goodcoder/sunminqi/C++-id-15633 --username=sunminqi

app:
	bcloud local --workspace=reuse --build=app -j2 /home/sunstrike/goodcoder/sunminqi/C++-id-15633 --username=sunminqi

ut:
	bcloud local --workspace=reuse --build=ut -j2 /home/sunstrike/goodcoder/sunminqi/C++-id-15633 --username=sunminqi

clean:
	bcloud local --build=clean /home/sunstrike/goodcoder/sunminqi/C++-id-15633 --username=sunminqi

cleanall:
	bcloud local --build=cleanall /home/sunstrike/goodcoder/sunminqi/C++-id-15633 --username=sunminqi

#update workspace,recreate the makefile
updateworkspace:
	bcloud local -j2 /home/sunstrike/goodcoder/sunminqi/C++-id-15633 --username=sunminqi

#recreate the makefile
updatemakefile:
	bcloud local --workspace=reuse --calcdep -j2 /home/sunstrike/goodcoder/sunminqi/C++-id-15633 --username=sunminqi 

#quick build
bcquick:
	cd /home/sunstrike && make -f /home/sunstrike/bc_out/bc_cache/goodcoder_sunminqi_C++-id-15633/bcloud.makefile_main -j2

test_parser:
	bcloud local --workspace=reuse --build=test_parser -j2    --username=sunminqi /home/sunstrike/goodcoder/sunminqi/C++-id-15633

parser:
	bcloud local --workspace=reuse --build=parser -j2    --username=sunminqi /home/sunstrike/goodcoder/sunminqi/C++-id-15633

