from __future__ import annotations

import logging
from pathlib import Path
from io import TextIOWrapper
import rich
from rich.prompt import FloatPrompt

from qseek.exporters.base import Exporter
from qseek.search import Search

from qseek.models.station import Station, Stations
from qseek.models.detection import EventDetection
from qseek.utils import PhaseDescription

logger = logging.getLogger(__name__)

class Velest(Exporter):
    """Crate a VELEST project folder for 1D velocity model estimation."""

    min_pick_semblance: float = 0.2
    pp:float = 0.3
    ps:float = 0.3
    tdiff:float = 2.5
    n_picks: dict[str, int] = {}
    n_events: int = 0

    async def export(self, rundir: Path, outdir: Path) -> Path:
        rich.print("Exporting qseek search to VELEST project folder")
        min_pick_semblance = FloatPrompt.ask("Minimum pick confidence", default=0.2)
        pp = FloatPrompt.ask("Minimum pick probability for P phase", default=0.3)
        ps = FloatPrompt.ask("Minimum pick probability for S phase", default=0.3)
        tdiff = FloatPrompt.ask("Maximum difference between model and observed arrival", default=2.5)
        self.min_pick_semblance = min_pick_semblance
        self.pp = pp
        self.ps = ps
        self.tdiff = tdiff
        self.n_picks= {"P":0,"S":0}

        outdir.mkdir()
        search = Search.load_rundir(rundir)
        phases = search.image_functions.get_phases()
        for phase in phases:
            if 'P' in phase:
                phaseP=phase
            if 'S' in phase:
                phaseS=phase
        if 'phaseP' not in locals() and 'phaseS' not in locals():
            print("Flail to find both P and S phase name." )
            return outdir
                      
        catalog = search.catalog  # noqa

        #export station file 
        stations=search.stations.stations
        station_file=outdir/"stations_velest.sta"
        self.export_station(stations=stations,filename=station_file)

        #export phase file
        phase_file = outdir / "phase_velest.pha"
        fp=open(phase_file,'w')
        neq=0
        for event in catalog:
            if event.semblance > min_pick_semblance:
                countp, counts=self.export_phase(event,fp,phaseP,phaseS,tdiff,pp,ps)
                self.n_picks['P']+=countp
                self.n_picks['S']+=counts
                neq+=1
        self.n_events=neq
        #export control file 
        params = {
            "reflat": search.octree.location.lat, # Reference Latitude
            "reflon": -search.octree.location.lon,  # Reference Longitude (should be negative!)
            "neqs": neq, # Number of earthquakes
            "distmax": 200, # Maximum distance from the station
            "zmin": -0.2, # Minimum depth
            "lowvelocity": 0, # Allow low velocity zones (0 or 1)
            "vthet": 1, # Damping parameter for the velocity
            "stathet": 0.1, # Damping parameter for the station
            "iuseelev": 0, # Use elevation (0 or 1)
            "iusestacorr": 0 # Use station corrections (0 or 1)
         }
        control_file=outdir / "velest.cmn"
        self.make_control_file(control_file,params, 1,'model.mod','phase_velest.pha','main.out','log.out','final.cnv','stacor.dat')

        #export velocity model file
        dep=search.ray_tracers.root[0].earthmodel.layered_model.profile("z") 
        vp=search.ray_tracers.root[0].earthmodel.layered_model.profile("vp") 
        vs=search.ray_tracers.root[0].earthmodel.layered_model.profile("vs") 
        dep_velest=[]
        vp_velest=[]
        vs_velest=[]
        for i, d in enumerate(dep):
            if float(d)/1000 not in dep_velest:
                dep_velest.append(float(d)/1000)
                vp_velest.append(float(vp[i])/1000)
                vs_velest.append(float(vs[i])/1000)      
        velmod_file=outdir / "model.mod"
        self.make_velmod_file(velmod_file,vp_velest,vs_velest,dep_velest )

        export_info = outdir / "export_info.json"
        export_info.write_text(self.model_dump_json(indent=2))
        print("Done!")
        return outdir
    
    @staticmethod
    def export_phase(
            event: EventDetection,
            fp:TextIOWrapper,
            phaseP:PhaseDescription,
            phaseS:PhaseDescription,
            tdiff:float,
            pp:float, 
            ps:float, 
            ) -> int:
        year=int(str(event.time.year)[-2::])
        # export phase catalog into txt file (for velest, format: ised=1)
        # tdiff: time diff tobs-tmodel threshold P,S 
        # pp,ps : minimun probability threshold P,S
        # there is quality weigth (0-4, 0 is best) in velest, here use probability to define it
        month=event.time.month
        day=event.time.day
        hour=event.time.hour
        min=event.time.minute
        sec=event.time.second+event.time.microsecond/1000000
        lat=event.effective_lat
        lon=event.effective_lon
        dep=event.depth/1000
        if event.magnitude !=None:
            mag=event.magnitude.average
        else:
            mag=0.0
        if lat<0:
            vsn='S'
            lat=abs(lat)
        else:
            vsn='N'
        if lon<0:
            vew='W'
            lon=abs(lon)
        else:
            vew='E'   
        fp.write("%2d%2d%2d %2d%2d %5.2f %7.4f%s %8.4f%s %7.2f  %5.2f\n"%(year,month,day,hour,min,sec,lat,vsn,lon,vew,dep,mag))
        countP=0
        countS=0
        for receiver in event.receivers.receivers:
            station=receiver.station
            if receiver.phase_arrivals[phaseP].observed !=None and receiver.phase_arrivals[phaseP].observed.detection_value >=pp:
                tpick=receiver.phase_arrivals[phaseP].observed.time-event.time
                tpick=tpick.total_seconds()
                dt=receiver.phase_arrivals[phaseP].traveltime_delay.total_seconds()
                if abs(dt)>tdiff:
                    continue
                if receiver.phase_arrivals[phaseP].observed.detection_value <0.4:
                    iwt=3
                elif receiver.phase_arrivals[phaseP].observed.detection_value <0.6:
                    iwt=2
                elif receiver.phase_arrivals[phaseP].observed.detection_value <0.8:
                    iwt=1
                else:
                    iwt=0
                phase='P'
                fp.write("  %-6s  %-1s   %1d  %6.2f\n"%(station,phase,iwt,tpick))
                countP+=1
            if receiver.phase_arrivals[phaseS].observed !=None and receiver.phase_arrivals[phaseS].observed.detection_value >=ps:
                tpick=receiver.phase_arrivals[phaseS].observed.time-event.time
                tpick=tpick.total_seconds()
                dt=receiver.phase_arrivals[phaseS].traveltime_delay.total_seconds()
                if abs(dt)>tdiff:
                    continue
                if receiver.phase_arrivals[phaseS].observed.detection_value <0.4:
                    iwt=3
                elif receiver.phase_arrivals[phaseS].observed.detection_value <0.6:
                    iwt=2
                elif receiver.phase_arrivals[phaseS].observed.detection_value <0.8:
                    iwt=1
                else:
                    iwt=0
                phase='S'
                fp.write("  %-6s  %-1s   %1d  %6.2f\n"%(station,phase,iwt,tpick))
                countS+=1
        if countP == 0 and countS == 0 :
            print("No good phases obesered for event%s, lower pick probability or increase pick confidence"%event.time)
        fp.write("\n")
        return countP , countS

    @staticmethod
    def export_station(
            stations: list[Station],
            filename:Path
            )-> None:
        fpout=open(filename,'w')
        fpout.write("(a6,f7.4,a1,1x,f8.4,a1,1x,i4,1x,i1,1x,i3,1x,f5.2,2x,f5.2)\n")
        p2=1
        p3=1
        v1=0.00
        v2=0.00
        for station in stations:
            lat=station.lat
            lon=station.lon
            sta=station.station
            elev=station.elevation
            if lat<0:
                vsn='S'
                lat=abs(lat)
            else:
                vsn='N'
            if lon<0:
                vew='W'
                lon=abs(lon)
            else:
                vew='E' 
            fpout.write("%-6s%7.4f%1s %8.4f%s %4d %1d %3d %5.2f  %5.2f\n"%(sta,lat,vsn,lon,vew,elev,p2,p3,v1,v2))
            p3+=1
        fpout.close()

    @staticmethod
    def make_control_file(filename, params,isingle,modname,phasefile,mainoutfile,outcheckfile,finalcnv,stacorfile):
        
        if isingle == 1:
            ittmax = 99
            invertratio = 0
        else:
            ittmax = 9
            invertratio = 3
        fp=open(filename,'w')
        fp.write("velest parameters are below, please modify according to their documents\n")
        #***  olat       olon   icoordsystem      zshift   itrial ztrial    ised
        fp.write("%s   %s      0            0.0      0     0.00      1\n"%(params["reflat"],params["reflon"]))
        #*** neqs   nshot   rotate
        fp.write("%s      0      0.0\n"%params["neqs"])
        #*** isingle   iresolcalc
        fp.write("%s     0\n"%isingle)
        #*** dmax    itopo    zmin     veladj    zadj   lowveloclay
        fp.write("%s  0      %s    0.20    5.00    %s\n"%(params["distmax"],params["zmin"],params["lowvelocity"] ))
        #*** nsp    swtfac   vpvs       nmod
        fp.write( "2      0.75      1.650        1\n")
        #***   othet   xythet    zthet    vthet   stathet
        fp.write( "0.01    0.01      0.01    %s     %s\n"%(params["vthet"],params["stathet"]))
        #*** nsinv   nshcor   nshfix     iuseelev    iusestacorr
        fp.write(  "1       0       0        %s        %s\n"%(params["iuseelev"],params["iusestacorr"]))
        #*** iturbo    icnvout   istaout   ismpout
        fp.write(  "1         1         2        0\n")
        #*** irayout   idrvout   ialeout   idspout   irflout   irfrout  iresout
        fp.write(  "0         0         0         0         0         0        0\n")
        #*** delmin   ittmax   invertratio
        fp.write(  "0.001   %s   %s\n"%(ittmax,invertratio))
        #*** Modelfile:
        fp.write(  "%s\n"%modname)
        #*** Stationfile:
        fp.write(  "stations_velest.sta\n")
        #*** Seismofile:
        fp.write(  " \n")                                                                             
        #*** File with region names:
        fp.write(  "regionsnamen.dat\n")
        #*** File with region coordinates:
        fp.write(  "regionskoord.dat\n")
        #*** File #1 with topo data:
        fp.write(  " \n")                                                                               
        #*** File #2 with topo data:
        fp.write(  " \n")                                                                               
        #*** File with Earthquake data (phase):
        fp.write(  "%s\n"%phasefile)
        #*** File with Shot data:
        fp.write(  " \n")                                                                             
        #*** Main print output file:
        fp.write( "%s\n"%mainoutfile)
        #*** File with single event locations:
        fp.write( "%s\n"%outcheckfile)
        #*** File with final hypocenters in *.cnv format:
        fp.write( "%s\n"%finalcnv)
        #*** File with new station corrections:
        fp.write( "%s\n"%stacorfile)
        fp.close()

    @staticmethod
    def make_velmod_file(modname,vp,vs,dep ):
        nlayer=len(dep)
        vdamp = 1.0
        fp=open(modname,'w')
        fp.write("initial 1D-model for velest\n")
        # the second line - indicate the number of layers for Vp
        fp.write("%3d        vel,depth,vdamp,phase (f5.2,5x,f7.2,2x,f7.3,3x,a1)\n"%nlayer)
        #vp model
        for i,v in enumerate(vp):
            fp.write("%5.2f     %7.2f  %7.3f\n"%(v,dep[i],vdamp))
        #vs model
        fp.write("%3d\n"%nlayer)
        for i,v in enumerate(vs):
            fp.write("%5.2f     %7.2f  %7.3f\n"%(v,dep[i],vdamp))
        fp.close()
    