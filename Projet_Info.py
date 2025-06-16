import csv
from math import atan, atan2, pi, sqrt
from os import read
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.optimize import curve_fit

# Function to read CSV files and return a dictionary of celestial body data

def lecture(files):
    """
    Reads CSV files containing x, y, z coordinates for each celestial body.
    
    Parameters:
        files (dict): Dictionary in the form {body_name: file_path}
        
    Returns:
        dict: Structure {body_name: {'x': np.array, 'y': np.array, 'z': np.array}}
    """
    body = {}
    for name, f in files.items():
        with open(f, 'r', newline='') as f:            
            reader = csv.reader(f)            
            data = list(reader)
            N = len(data)
            x = np.zeros(N)
            y = np.zeros(N)
            z = np.zeros(N)
            for i, cologne in enumerate(data):
                # Columns 2, 3, 4 are expected to contain x, y, z coordinates
                x[i] = float(cologne[2])
                y[i] = float(cologne[3])
                z[i] = float(cologne[4])
            body[name] = {'x': x, 'y': y, 'z': z}
            
    return body


# Load trajectory data for each celestial body from CSV files

celestial_body = lecture({   
    'voy1': 'voy1.csv',
    'Jupiter':'jup.csv',
    'Saturn':'Saturn.csv',
    'voy2':'voy2.csv'
})

# Set up the 3D plot

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')

# Define axis limits based on the range of all coordinate data

all_x = np.concatenate([body['x'] for body in celestial_body.values()])
all_y = np.concatenate([body['y'] for body in celestial_body.values()])
all_z = np.concatenate([body['z'] for body in celestial_body.values()])
ax.set_xlim(np.min(all_x), np.max(all_x))
ax.set_ylim(np.min(all_y), np.max(all_y))
ax.set_zlim(np.min(all_z), np.max(all_z))

# Initialize containers for plotting points and trajectories

points = {}
lignes = {}
colors = {'voy1': 'red','Jupiter':'green','Saturn':'yellow','voy2':'purple'}

# Initialize scatter points and trajectory lines for each body

for name, body in celestial_body.items():
    # Draw current position as a scatter point
    points[name] = ax.scatter(
        body['x'][0:1], body['y'][0:1], body['z'][0:1],
        c=colors[name], marker='o', s=50, label=name
    )
    # Draw trajectory line (will be updated during animation)
    lignes[name], = ax.plot([], [], [], color=colors[name], alpha=0.5, lw=1)
    # Add legend to identify each body
ax.legend()

# ---------------------------------------------------------------
# ANIMATION FUNCTION
# ---------------------------------------------------------------

def update(f):
    
    """
    Updates the positions and trajectories of the celestial bodies at each frame.
    
    Parameters:
        f (int): Current frame number
    """    
    
    for names, body in celestial_body.items():
        # Update current position (scatter point)
        points[names]._offsets3d = (
            [body['x'][f]],
            [body['y'][f]],
            [body['z'][f]]
        )
        
        # Update trajectory (line) with the last known positions
        lignes[names].set_data(body['x'][:f+1], body['y'][:f+1])
        lignes[names].set_3d_properties(body['z'][:f+1])
        
    ax.set_title(f"Time Step: {f}")
    return list(points.values()) + list(lignes.values())

# ---------------------------------------------------------------
# ANIMATION CONFIGURATION
# ---------------------------------------------------------------

f_max = len(celestial_body['voy1']['x'])    # Total number of frames (based on Voyager 1)
pas = 25  # Reduce number of frames for smoother animation

ani = FuncAnimation(
    fig,
    update,
    frames=range(0, f_max, pas),
    interval=20,    # Milliseconds between frames
    blit=False      # blit must be False for 3D plots
)

plt.show()

# ---------------------------------------------------------------
# TRAJECTORY ANALYSIS - LINEAR / SINUSOIDAL MODELING
# ---------------------------------------------------------------

ncord = 4000    # Start of analysis (skip early data due to complex movement of Voyager 1)

def create_coord(b):
    """
    Extracts and shifts a segment of data for modeling.

    Parameters:
        b (array-like): Original coordinate array (e.g., x or y or z)

    Returns:
        tuple: (segment of coordinate data, corresponding time steps)
    """
    N = len(b)
    x_lin = np.zeros(N-ncord)
    delta_T = np.zeros(N-ncord)
    for i in range(N):
        x_lin[i-ncord] = b[i]
        delta_T[i-ncord] = (i-ncord)*10     # 10-hour intervals between points
    return (x_lin ,delta_T)

# ---------------------------------------------------------------
# BASIC SINUSOIDAL MODELING WITH CURVE_FIT
# ---------------------------------------------------------------


def sine_func(x, A, B, C, D):
    
    """
    Basic sinusoidal model: A * sin(Bx + C) + D

    Parameters:
        x (array-like): Time or input variable
        A, B, C, D (float): Model parameters

    Returns:
        array: Modeled output values
    """
    
    return A * np.sin(B * x + C) + D

def r1(x, A, B, C, D):
    
    """
    Residuals: difference between data and the model.

    Parameters:
        x (array-like): Actual data
        A, B, C, D (float): Model parameters

    Returns:
        array: Residual values
    """   
    
    return x- sine_func(x, A, B, C, D)


# ---------------------------------------------------------------
# GAUSS-NEWTON JACOBIAN CALCULATION
# ---------------------------------------------------------------


def JacCalc1(x, A_guess, B_guess, C_guess, D_guess):
    
    """
    Computes the Jacobian matrix used in Gauss-Newton optimization
    for the sinusoidal model.

    Parameters:
        x (np.array): Time values
        A_guess, B_guess, C_guess, D_guess (float): Initial parameter guesses

    Returns:
        Delta (np.array): Parameter update vector
    """
    
    theta = B_guess * x + C_guess
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    
    # Construct Jacobian matrix manually

    J = np.array([
        -sin_theta, # Derivative with respect to A                   
        -A_guess * x * cos_theta,  # Derivative with respect to B    
        -A_guess * cos_theta,   # Derivative with respect to C                 
    ])

    # Residual between data and model 

    residual = r1(x, A_guess, B_guess, C_guess, D_guess)
    
    # Solve the normal equation

    Delta = np.linalg.lstsq(J @ J.T, J @ residual)[0]
    return Delta


# ---------------------------------------------------------------
# SINUSOIDAL FITTING USING CURVE_FIT
# ---------------------------------------------------------------



def sininterpol(x,p0):
    
    """
    Fits a sinusoidal model to the data using scipy's curve_fit.

    Parameters:
        x (np.array): Observed data
        p0 (tuple): Initial parameter guess (A, B, C, D)

    Returns:
        tuple: Optimal parameters (A, B, C, D)
    """
    
    params, params_covariance = curve_fit(
        sine_func, 
        delta_T, 
        x, 
        p0=p0, 
        maxfev=10000
    )
    Ax, Bx, Cx, Dx = params    
    return Ax, Bx, Cx, Dx   # Retourne (A, B, C, D) optimaux

# ---------------------------------------------------------------
# APPLY TO VOYAGER 1 DATA
# ---------------------------------------------------------------

(xlim,delta_T,ylim,zlim) = ( 
create_coord(celestial_body['voy1']['x'])[0],
create_coord(celestial_body['voy1']['x'])[1],
create_coord(celestial_body['voy1']['y'])[0],
create_coord(celestial_body['voy1']['z'])[0]
)

# ---------------------------------------------------------------
# UTILITIES FOR MODELING & PLOTTING
# ---------------------------------------------------------------

def avg(v):
    
    """
    Computes the average (mean) of a list or array.
    
    Parameters:
        v (array-like): Input values

    Returns:
        float: Mean value
    """
    
         
    return sum(v) / len(v)

def f(xx,aa,bb):
    
    """
    Simple linear model: f(x) = a·x + b

    Parameters:
        xx (array-like): Input time points
        aa, bb (float): Slope and intercept

    Returns:
        np.array: Linear function values
    """

    
    return np.array([aa * x + bb for x in xx])

def plot2f(xx,yy,xa,ya):    
    
    """
    Plots observed vs. modeled values (e.g. real vs predicted positions)

    Parameters:
        xx, yy (array-like): Observed data
        xa, ya (array-like): Model predictions
    """
    
    plt.figure(1,figsize=(6,4 ),dpi =120 )
    plt.plot(xa,ya,'--r', label='Model')
    
    plt.scatter(xx, yy,color='blue',marker='o', label='Data')
    plt.title('Observed Position vs Model Prediction')
    plt.xlabel('Time since 1977 (in hours)')
    plt.ylabel('Position (km)')
    plt.legend()
    plt.grid()
    plt.show()
    return


# ---------------------------------------------------------------
# CARTESIAN TO EQUATORIAL COORDINATE CONVERSION
# ---------------------------------------------------------------


def cartToPol(x,y,z):
    
    """
    Converts Cartesian coordinates (x, y, z) to Equatorial coordinates:
    Right Ascension (RA) and Declination (Dec)

    Parameters:
        x, y, z (float): Cartesian coordinates

    Returns:
        tuple: (RA, Dec) in radians

    Also prints RA and Dec in HH:MM:SS and DD:MM:SS format
    """
    
    Ra = atan2(y,x) # RA in radians (0 to 2 pi)
    Dec = atan(z/(sqrt(x**2 + y**2)))
    if(Ra < 0):
        Ra = Ra+ 2 * pi
        
    # Convert to degrees-minutes-seconds (DMS) for Declination
    Decdeg = Dec *180/pi    
    Decm = (Decdeg - int(Decdeg))*60
    Decs = (Decm- int(Decm))*60
    
    # Convert to hours-minutes-seconds (HMS) for RA
    Rah = Ra*12/pi
    Ram = (Rah - int(Rah))*60
    Ras = (Ram - int(Ram))*60
    
    print('Ra = ' , int(Rah) , 'H' ,int(Ram),'m',int(Ras),'s'  )
    print('Dec = ' , int(Decdeg),'deg',int(Decm),'m',int(Decs),'s')
    return (Ra,Dec)

# ---------------------------------------------------------------
# TIME VECTOR (from September 6, 1977 to May 16, 2025)
# ---------------------------------------------------------------

N1 = 41806      # 418,060 hours → corresponds to ~47.7 years, from 6 september 1977 to 16 may 2025
T1 = np.zeros(N1)

# Constructing the time vector: one value every 10 hours
for i in range (N1):
    
    T1[i] = 10*i 

# ---------------------------------------------------------------
# LINEAR MODEL FITTING USING LEAST SQUARES (x = a*t + b)
# ---------------------------------------------------------------

# x-component

aax = np.dot(delta_T-avg(delta_T),xlim-avg(xlim)) / np.dot(delta_T - avg(delta_T),delta_T - avg(delta_T))
bbx = avg(xlim) - aax* avg(delta_T)

# y-component
aay = np.dot(delta_T-avg(delta_T),ylim-avg(ylim)) / np.dot(delta_T - avg(delta_T),delta_T - avg(delta_T))
bby = avg(ylim) - aay* avg(delta_T)

# y-component
aaz = np.dot(delta_T-avg(delta_T),zlim-avg(zlim)) / np.dot(delta_T - avg(delta_T),delta_T - avg(delta_T))
bbz = avg(zlim) - aaz* avg(delta_T)

# ---------------------------------------------------------------
# PLOT OBSERVED TRAJECTORIES AND LINEAR REGRESSION MODELS
# ---------------------------------------------------------------

# Each call compares observed position with the linear model
plot2f(delta_T,xlim,T1,f(T1,aax,bbx))
plot2f(delta_T,ylim,T1,f(T1,aay,bby))
plot2f(delta_T,zlim,T1,f(T1,aaz,bbz))

# ---------------------------------------------------------------
# COMPUTING FINAL POSITION FROM THE LINEAR MODEL
# ---------------------------------------------------------------

nf = len(f(T1,aaz,bbz))     # Total time steps in prediction

# Final extrapolated position of Voyager 1
xf = f(T1,aax,bbx)[nf-1] 
yf = f(T1,aay,bby)[nf-1]
zf = f(T1,aaz,bbz)[nf-1] 

# Display results
print('Predicted Equatorial Coordinates (radians) from Linear Model:')
print('RA, Dec:', cartToPol(xf, yf, zf))
print('Distance from Earth:', sqrt(xf**2 + yf**2 + zf**2) / 149_697_870.7, 'AU')  # Convert km to AU

# ---------------------------------------------------------------
# HYBRID MODEL: LINEAR TREND + SINUSOIDAL RESIDUALS
# ---------------------------------------------------------------

def sinex_func(x,a, A, B, C, b):
    """
    Hybrid model combining a linear trend and a sinusoidal component:
    f(x) = a*x + b + A*sin(B*x + C)

    Parameters:
        x (array-like): Time values
        a, b (float): Linear coefficients
        A, B, C (float): Sinusoidal parameters

    Returns:
        np.array: Modeled values
    """
    return A * np.sin(B * x + C) + a*x + b

def r2(x,a, A, B, C, b):
    """
    Computes residuals for the hybrid model.

    Returns:
        np.array: Difference between actual and predicted values
    """
    return x- sinex_func(x,a, A, B, C, b)


def JacCalc2(x, a, A_guess, B_guess, C_guess, b):
    
    """
    Computes the Jacobian matrix for Gauss-Newton optimization
    of the hybrid model: linear + sinusoidal.

    Parameters:
        x (np.array): Time values
        a, b (float): Linear trend parameters
        A_guess, B_guess, C_guess (float): Sinusoidal guesses

    Returns:
        np.array: Parameter update vector (ΔA, ΔB, ΔC)
    """
    
    theta = B_guess * x + C_guess
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    
    J = np.array([
        -sin_theta,                 # ∂/∂A                    
        -A_guess * x * cos_theta,   # ∂/∂B         
        -A_guess * cos_theta        # ∂/∂C   
    ])
    residual = r2(x,a, A_guess, B_guess, C_guess, b)
    
    # Solve the normal equations to get parameter updates
    Delta = np.linalg.lstsq(J @ J.T, J @ residual)[0]
    
    return Delta

def sinxinterpol(x,A,B,C):
    
    """
    Performs a two-stage fitting process:
    1. Fits a linear trend (least squares)
    2. Fits a sinusoidal function to the residuals
    
    Parameters:
        x (np.array): Observed data (position)
        A, B, C (float): Initial guesses for the sinusoidal parameters
        
    Returns:
        Tuple: (a, A, B, C, b) where
               - a, b: linear coefficients
               - A, B, C: sinusoidal parameters
    """
    
    # Step 1: Linear trend fitting
    
    a = np.dot(delta_T-avg(delta_T),x-avg(x)) / np.dot(delta_T - avg(delta_T),delta_T - avg(delta_T))
    b = avg(x) - a* avg(delta_T) 
    
    # Step 2: Compute residuals after removing trend
    xdata = f(delta_T,a,b) - x
    
    # Initial sinusoidal parameter guesses
    A_guessx = A
    C_guessx = C
    B_guessx =B
    
    # Diagnostic plot of residual vs initial sinusoidal guess
    plot2f(delta_T,xdata,delta_T,sine_func(delta_T,A_guessx,B_guessx,C_guessx,0))
    
    # Iterative Gauss-Newton optimization (30 iterations)
    betha = [A_guessx,B_guessx,C_guessx]
    Delta = JacCalc2(x,a, A_guessx, B_guessx, C_guessx, b)
    betha = betha - Delta
    for i in range (30):
        A_guessx, B_guessx, C_guessx = betha    
        betha = betha - JacCalc2(x,a, A_guessx, B_guessx, C_guessx, b)
    Ax, Bx, Cx = betha
    return a,Ax, Bx, Cx, b  # return (a, A, B, C, b)
    
# Fit hybrid model on the 3 axes with tuned initial parameters
(aax,Ax, Bx, Cx, bbx) = sinxinterpol(xlim,152850000,7.115 * 10**(-4),2.7 * 10**3,)
(aay,Ay, By, Cy, bby) = sinxinterpol(ylim,138650000,1.185506 * 10**(-3),-1.3 * 10**3)
(aaz,Az, Bz, Cz, bbz) = sinxinterpol(zlim,121400000/2,1.3368*10**(-3),700)

# Visual validation of hybrid fit on each axis
plot2f(delta_T,xlim,T1,sinex_func(T1,aax, Ax, Bx, Cx, bbx))
plot2f(delta_T,ylim,T1,sinex_func(T1,aay, Ay, By, Cy, bby))
plot2f(delta_T,zlim,T1,sinex_func(T1,aaz, Az, Bz, Cz, bbz))

# Final predicted position using hybrid model
xf = sinex_func(T1,aax, Ax, Bx, Cx, bbx)[nf-1]
yf = sinex_func(T1,aay, Ay, By, Cy, bby)[nf-1]
zf = sinex_func(T1,aaz, Az, Bz, Cz, bbz)[nf-1]

# Conversion to equatorial coordinates and distance in AU

print('Predicted RA/Dec (radians) from hybrid model, Voyager 1:')
print('→', cartToPol(xf, yf, zf))
print('→ Distance from Earth:', sqrt(xf**2 + yf**2 + zf**2) / 149_697_870.7, 'AU')

# Reset data for Jupiter

ncord = 0 # No need to adjust time range manually

# Load Jupiter's Cartesian coordinates
(xlim,delta_T,ylim,zlim) = (
    create_coord(celestial_body['Jupiter']['x'])[0],
    create_coord(celestial_body['Jupiter']['x'])[1],
    create_coord(celestial_body['Jupiter']['y'])[0],
    create_coord(celestial_body['Jupiter']['z'])[0]
)

N1 = 41806  # Number of time steps (hours since reference point)

# === Initial guesses for the simple sinusoidal fit ===
A_guessx = 937500000
B_guessx = 5.99e-5
C_guessx = 83400
D_guessx = 0
p0 = [A_guessx,B_guessx,C_guessx,D_guessx]

# === Simple sinusoidal interpolation: F(x) = A * sin(Bx + C) + D ===
# Plotting initial guess to visualize estimation accuracy
plot2f(delta_T,xlim,delta_T,sine_func(delta_T, A_guessx,B_guessx,C_guessx,D_guessx))

# Curve fitting for each Cartesian coordinate using the sine model
(Ax, Bx, Cx, Dx) = sininterpol(xlim,p0)
(Ay, By, Cy, Dy) = sininterpol(ylim,p0)
(Az, Bz, Cz, Dz) = sininterpol(zlim,p0)

# Plotting the fitted results for validation
plot2f(delta_T,xlim,T1,sine_func(T1, Ax, Bx, Cx, Dx))
plot2f(delta_T,ylim,T1,sine_func(T1, Ay, By, Cy, Dy))
plot2f(delta_T,zlim,T1,sine_func(T1, Az, Bz, Cz, Dz))

# Extracting final Cartesian coordinates from the fitted functions
xf = sine_func(T1, Ax, Bx, Cx, Dx)[nf-1]
yf = sine_func(T1, Ay, By, Cy, Dy)[nf-1]
zf = sine_func(T1, Az, Bz, Cz, Dz)[nf-1]

# Displaying final coordinates and radial distance in Astronomical Units (AU)
print("Spherical coordinates using y = A·sin(Bx + C) + D model for Jupiter:", cartToPol(xf, yf, zf))
print("Distance from Earth in Astronomical Units (AU):", sqrt(xf**2 + yf**2 + zf**2) / 149_697_870.7, "AU")

# === Double sinusoidal model: y = A·sin(Bx + C) + D·sin(Ex + F) + G ===
def sine2_func(x, A, B, C, D ,E ,F ,G):    
    """Two-component sinusoidal model."""    
    return A * np.sin(B * x + C) + D * np.sin(E * x + F) + G

def sin2interpol(x,p0):    
    """Fit two-component sine model to the data with initial parameter estimates."""   
    params, params_covariance = curve_fit(
        sine2_func, 
        delta_T, 
        x, 
        p0=p0, 
        maxfev=10000
    )    
    Ax, Bx, Cx, Dx ,Ex ,Fx ,Gx = params 
    return Ax, Bx, Cx, Dx, Ex, Fx, Gx

# === Curve fitting for Jupiter’s x and y coordinates ===
A_guessx = 770500000
B_guessx = 5.99e-5
C_guessx = 83400
D_guessx = 157500000
E_guessx = 6.82e-4
F_guessx = 0
G_guessx = 0
p0 = [A_guessx,B_guessx,C_guessx,D_guessx,E_guessx,F_guessx,G_guessx]

(Ax, Bx, Cx, Dx, Ex, Fx, Gx) = sin2interpol(xlim,p0)
(Ay, By, Cy, Dy, Ey, Fy, Gy) = sin2interpol(ylim,p0)

# === Separate fitting for Jupiter’s z coordinate (fine-tuned initial guess) ===
A_guessx = 300500000
B_guessx = 6.053e-5
C_guessx = 1.082e+5
D_guessx = 54600000
E_guessx = 7.39e-4
F_guessx = 0
G_guessx = 0
p0 = [A_guessx,B_guessx,C_guessx,D_guessx,E_guessx,F_guessx,G_guessx]

(Az, Bz, Cz, Dz, Ez, Fz, Gz) = sin2interpol(zlim,p0)

# === Plotting the final fitted curves for all three coordinates ===
plot2f(delta_T,xlim,T1,sine2_func(T1, Ax, Bx, Cx, Dx, Ex, Fx, Gx))
plot2f(delta_T,ylim,T1,sine2_func(T1, Ay, By, Cy, Dy, Ey, Fy, Gy))
plot2f(delta_T,zlim,T1,sine2_func(T1, Az, Bz, Cz, Dz, Ez, Fz, Gz))

# Extracting final Cartesian positions from fitted double sine functions
xf = sine2_func(T1, Ax, Bx, Cx, Dx, Ex, Fx, Gx)[nf-1]
yf = sine2_func(T1, Ay, By, Cy, Dy, Ey, Fy, Gy)[nf-1]
zf = sine2_func(T1, Az, Bz, Cz, Dz, Ez, Fz, Gz)[nf-1]

# Final output in spherical coordinates and distance in AU

print("Spherical coordinates using y = A·sin(Bx + C) + D·sin(Ex + F) + G model for Jupiter:", cartToPol(xf, yf, zf))
print("Distance in Astronomical Units (AU):", sqrt(xf**2 + yf**2 + zf**2) / 149_697_870.7, "AU")