from __future__ import print_function, division
import numpy as np
import datetime as dt
from pydap.client import open_url
from netCDF4 import Dataset
from .great_circle_tools import gc_bear, gc_dist
from scipy.signal import convolve2d
from .old_gradient import gradient
from .uhr_tools import select_amb

RAPIDSCAT_INIT_DT = dt.datetime(1999, 1, 1)
ASCAT_INIT_DT = dt.datetime(1990, 1, 1)


class _Scatterometer(object):

    def __init__(self, url, pydap=True):
        if pydap:
            self.url = open_url(url)
        else:
            self.url = Dataset(url)

    def _compute_uv(self, ws, wd):
        return ws * np.sin(np.deg2rad(wd)), ws * np.cos(np.deg2rad(wd))

    def _transform_uv(self, u, v, angle, rad=False):
        """
        u = U component in Earth-centric coordinate system
        v = V component in Earth-centric coordinate system
        angle = Rotation angle of swath (defined CCW from 90-deg East)
        rad = True if angle is in radians, False if degrees
        """
        if not rad:
            angle = np.deg2rad(angle)
        up = u * np.cos(angle) + v * np.sin(angle)
        vp = v * np.cos(angle) - u * np.sin(angle)
        return up, vp

    def _get_rotation_angle(self, lat, lon):
        swbear = []
        for ix in range(np.shape(lat)[0]):
            lat1 = lat[ix]
            lon1 = lon[ix]
            bear = []
            for j in range(len(lat1)):
                if j > 0:
                    if lat1[j] >= -90 and lat1[j] <= 90 and \
                            lon1[j] >= -360 and lon1[j] <= 360:
                        bear.append(gc_bear(lat1[0], lon1[0],
                                            lat1[j], lon1[j]))
            # May fail if no good lat/lon
            swbear.append(np.mean(bear))
        # Setting 0 point as due East for _transform_uv
        return 90.0 - np.array(swbear)

    def _get_uprime_vprime(self, u, v, lat, lon):
        rot = self._get_rotation_angle(lat, lon)
        up = []
        vp = []
        for i in range(len(rot)):
            up1, vp1 = self._transform_uv(
                u[i].filled(fill_value=1e20),
                v[i].filled(fill_value=1e20), rot[i])
            up.append(up1)
            vp.append(vp1)
        up = np.array(up)
        vp = np.array(vp)
        return np.ma.masked_where(np.logical_or(up > 1000, up < -1000), up), \
            np.ma.masked_where(np.logical_or(vp > 1000, vp < -1000), vp)

    def _calc_div_finite_diff(self, u, v, lat, lon, smooth=1, res=12500.0,
                              variable_res=False):
        au, av = self._get_uprime_vprime(u, v, lat, lon)
        dxx = np.zeros((au.shape[0], au.shape[1]))
        dyy = np.zeros((av.shape[0], av.shape[1]))
        divg = np.zeros((au.shape[0], au.shape[1]))
        if smooth is not None:
            matrix = self._get_matrix(smooth)
            smu = convolve2d(au, matrix, mode='same', boundary='symm',
                             fillvalue=-999.0)
            smv = convolve2d(av, matrix, mode='same', boundary='symm',
                             fillvalue=-999.0)
        else:
            smu = 1.0 * au
            smv = 1.0 * av
        if variable_res:
            dxx, dyy = self._get_dx_dy_from_lat_lon(lat, lon)
        else:
            dxx[:] = res
            dyy[:] = res
        dqu_dy, dqu_dx = gradient(smu[:, :], dyy, dxx)
        dqv_dy, dqv_dx = gradient(smv[:, :], dyy, dxx)
        divg = dqu_dx + dqv_dy
        divg = np.ma.masked_where(np.logical_or(au.mask, av.mask), divg)
        divg.mask = np.logical_or(np.logical_or(
            divg > 1, divg < -1), divg.mask)
        windgrad = np.sqrt((dqu_dx + dqv_dx)**2 + (dqu_dy + dqv_dy)**2)
        windgrad = np.ma.masked_where(
            np.logical_or(au.mask, av.mask), windgrad)
        windgrad.mask = np.logical_or(np.logical_or(
            windgrad > 1, windgrad < -1), windgrad.mask)
        return divg, windgrad

    def _get_matrix(self, smooth):
        x = smooth
        return np.ones((x, x), dtype='float') / x**2

    def _get_dx_dy_from_lat_lon(self, lat, lon):
        shp = np.shape(lat)
        dy = np.zeros_like(lat)
        dx = 0.0 * dy
        for j in range(shp[0]-1):
            for i in range(shp[1]-1):
                dy[j, i] = gc_dist(lat[j, i], lon[j, i],
                                   lat[j+1, i], lon[j+1, i])
                dx[j, i] = gc_dist(lat[j, i], lon[j, i],
                                   lat[j, i+1], lon[j, i+1])
        dy[-1, :] = dy[-2, :]
        dx[-1, :] = dx[-2, :]
        dy[:, -1] = dy[:, -2]
        dx[:, -1] = dx[:, -2]
        return 1000.0 * dx, 1000.0 * dy


class Ascat(_Scatterometer):

    def __init__(self, url, pydap=True, scale_coord=1e-5,
                 scale_ws=0.01, scale_wd=0.1):
        _Scatterometer.__init__(self, url, pydap=pydap)
        self.populate_attributes(scale_coord=scale_coord, scale_ws=scale_ws,
                                 scale_wd=scale_wd)
        if not pydap:
            self.url.close()

    def populate_attributes(self, scale_coord=1e-5, scale_ws=0.01,
                            scale_wd=0.1):
        self.data = {}
        self.data['left'] = {}
        self.data['right'] = {}
        self.data['datetime'] = []
        ws = np.array(self.url['wind_speed']) * scale_ws
        wd = np.array(self.url['wind_dir']) * scale_wd
        self.index = int(np.shape(ws)[1] / 2)
        cond = np.logical_and(ws >= 0, wd >= 0)
        self.data['left']['wind_speed'] = np.ma.masked_where(~cond, ws)[
            :, :self.index]
        self.data['right']['wind_speed'] = np.ma.masked_where(~cond, ws)[
            :, self.index:]
        self.data['left']['wind_dir'] = np.ma.masked_where(~cond, wd)[
            :, :self.index]
        self.data['right']['wind_dir'] = np.ma.masked_where(~cond, wd)[
            :, self.index:]
        for key in ['left', 'right']:
            self.data[key]['u'], self.data[key]['v'] = self._compute_uv(
                self.data[key]['wind_speed'], self.data[key]['wind_dir'])
        self.data['left']['longitude'] = np.array(self.url['lon'])[
            :, :self.index] * scale_coord
        self.data['right']['longitude'] = np.array(self.url['lon'])[
            :, self.index:] * scale_coord
        self.data['left']['latitude'] = np.array(self.url['lat'])[
            :, :self.index] * scale_coord
        self.data['right']['latitude'] = np.array(self.url['lat'])[
            :, self.index:] * scale_coord
        self.data['left']['flags'] = np.array(self.url['wvc_quality_flag'])[
            :, :self.index]
        self.data['right']['flags'] = np.array(self.url['wvc_quality_flag'])[
            :, self.index:]
        # Time is 2D variable but is same across swath, save only first index
        for i in range(np.shape(self.url['time'])[0]):
            self.data['datetime'].append(
                ASCAT_INIT_DT + dt.timedelta(
                    seconds=float(self.url['time'][i, 0])))
        self.data['datetime'] = np.array(self.data['datetime'])

    def calc_divergence(self, smooth=1, res=12500.0, finite_diff=True):
        for key in ['left', 'right']:
            if finite_diff:
                divg, windgrad = self._calc_div_finite_diff(
                    self.data[key]['u'], self.data[key]['v'],
                    self.data[key]['latitude'], self.data[key]['longitude'],
                    smooth=smooth, res=res)
                self.data[key]['div'] = divg
                self.data[key]['grad'] = windgrad


class RapidScat(_Scatterometer):
    """
    Class for ingesting and analyzing RapidScat granules.
    """
    def __init__(self, url, pydap=True, scale_coord=1.0,
                 scale_ws=1.0, scale_wd=1.0):
        _Scatterometer.__init__(self, url, pydap=pydap)
        self.populate_attributes(scale_coord=scale_coord, scale_ws=scale_ws,
                                 scale_wd=scale_wd)
        if not pydap:
            self.url.close()

    def populate_attributes(self, scale_coord=1.0, scale_ws=1.0,
                            scale_wd=1.0):
        """
        Parameters
        ----------
        scale_coord : float, not used
        scale_ws : float, not used
        """
        self.data = {}
        self.data['longitude'] = np.array(self.url['lon'])
        self.data['latitude'] = np.array(self.url['lat'])
        self.data['rain_impact'] = np.array(self.url['rain_impact'])
        self.data['flags'] = np.array(self.url['flags'])
        ws = np.array(self.url['retrieved_wind_speed'])
        wd = np.array(self.url['retrieved_wind_direction'])
        cond = np.logical_and(wd > -9999, ws > -9999)
        self.data['wind_speed'] = np.ma.masked_where(~cond, ws)
        self.data['wind_dir'] = np.ma.masked_where(~cond, wd)
        self.data['u'], self.data['v'] = self._compute_uv(
            self.data['wind_speed'], self.data['wind_dir'])
        self.data['datetime'] = []
        for t in self.url['time']:
            self.data['datetime'].append(
                RAPIDSCAT_INIT_DT + dt.timedelta(seconds=t))
        self.data['datetime'] = np.array(self.data['datetime'])

    def calc_divergence(self, smooth=1, res=12500.0, finite_diff=True):
        if finite_diff:
            divg, windgrad = self._calc_div_finite_diff(
                self.data['u'], self.data['v'],
                self.data['latitude'], self.data['longitude'],
                smooth=smooth, res=res)
            self.data['div'] = divg
            self.data['grad'] = windgrad
        else:
            print('Other divergence methods not yet supported,',
                  'use finite_diff=True for now')


class UHR(_Scatterometer):

    def __init__(self, url):
        _Scatterometer.__init__(self, url, pydap=False)
        self.populate_attributes()
        self.url.close()

    def populate_attributes(self):
        self.data = {}
        self.data['longitude'] = np.array(self.url['lon'])
        self.data['latitude'] = np.array(self.url['lat'])
        asel = np.array(self.url['ambiguity_select'])
        ws = select_amb(np.array(self.url['wspeeds']), asel)
        wd = select_amb(np.array(self.url['wdirs']), asel)
        self.data['wind_speed'] = ws
        self.data['wind_dir'] = wd
        self.data['u'], self.data['v'] = self._compute_uv(
            self.data['wind_speed'], self.data['wind_dir'])

    def calc_divergence(self, smooth=None, res=12500.0, finite_diff=True):
        if finite_diff:
            divg, windgrad = self._calc_div_finite_diff(
                self.data['u'], self.data['v'],
                self.data['latitude'], self.data['longitude'],
                smooth=smooth, res=res, variable_res=True)
            self.data['div'] = divg
            self.data['grad'] = windgrad
        else:
            print('Other divergence methods not yet supported,',
                  'use finite_diff=True for now')
