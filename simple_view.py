# Copyright (c) 2008-2026 Optris GmbH & Co. KG

"""
Simple View Python3 example. 

2025-02-19 
"""

import sys
import threading
import cv2
import numpy as np
import optris.otcsdk as otc


class ImagerShow(otc.IRImagerClient):
    """
    A more feature rich implementation of an IRImagerClient that converts thermal frames to false color images and
    displays them.

    An IRImager acts as an observer to an IRImager implementation that retrieves and processes thermal data from
    Optris thermal cameras.
    """

    # Settings for the displayed fonts
    FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SIZE = 0.4

    # OpenCV colors (BGR)
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    RED   = (0, 0, 255)
    GREEN = (0, 255, 0)
    BLUE  = (255, 0, 0)


    def __init__(self, serial_number):
        """
        Constructor.
        """
        # Required to initialize base class
        super().__init__()

        # The factory is implemented as a Singleton. Therefore, you have to call getInstance()
        # first before you can create an IRImager object.
        #
        # The native implementation allows you to access the thermal data of cameras connected
        # via USB or Ethernet.
        self._imager = otc.IRImagerFactory.getInstance().create('native')

        # Register to receive updates via callbacks
        self._imager.addClient(self)

        # Establish a connection to the camera with the provided serial number
        self._imager.connect(serial_number)

        # Get a list of the available operation modes. Each operation mode is a valid combination of optics,
        # temperature ranges and video formats. The returned container is always sorted in the same way. The
        # availability of certain operation modes can depend on the used connection type (USB, Ethernet) to
        # device (different video formats).
        self._operation_modes   = self._imager.getOperationModes()
        self._active_mode_index = self._imager.getActiveOperationMode().getIndex()

        # Create an imager builder that converts thermal frames to false color images
        # The color format BGR is required because of OpenCV uses this pixel color oder by default
        self._builder = otc.ImageBuilder(colorFormat=otc.ColorFormat_BGR, widthAlignment=otc.WidthAlignment_OneByte)

        # The image grabbing and processing is done in a separate thread while the rendering of the
        # resulting image will be done in the main thread
        self._frame_event_lock    = threading.Lock()
        self._frame_event         = otc.ThermalFrame()
        self._frame_event_updated = False

        self._fps = otc.FramerateCounter(100)


    def __del__(self):
        """
        Destructor.
        """
        # Ensure the client removes itself before destruction to avoid memory access issues
        self._imager.removeClient(self)


    def run(self):
        """
        Main run method.
        """
        # Run the imager processing loop in a dedicated thread
        self._imager.runAsync()

        fps = 0.
        frame_event = None
        
        # Thermal frame to false color image conversion and rendering loop
        while self._imager.isRunning():
            # Get the latest thermal frame if there is one
            with self._frame_event_lock:
                if not self._frame_event_updated or self._frame_event.thermalFrame.isEmpty():
                  continue

                frame_event = self._frame_event
                fps = self._fps.getFps()
                self._frame_event_updated = False

            # Generate the false color image
            self._builder.setThermalFrame(thermalFrame=frame_event.thermalFrame)
            self._builder.convertTemperatureToPaletteImage()

            # Copy the image data to an empty NumPy array...
            image = np.empty((self._builder.getHeight(), self._builder.getWidth(), 3), dtype=np.uint8)
            self._builder.copyImageDataTo(image)

            # Inscribes a legend and hot and cold spots into the false color image
            self.drawOverlay(image, fps, frame_event.meta.getFlagState())
            self.calculateAndDrawTemperatureRegions(image)

            # ...and display it
            cv2.imshow('Optris Imager - {} (S/N {})'.format(otc.deviceTypeToString(self._imager.getDeviceType()), self._imager.getSerialNumber()), image)

            # Check for keyboard inputs indicating that the user wants to quit by pressing the q key
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
               break
            elif key == ord('r'):
                try:
                  self._imager.forceFlagEvent()
                except otc.SDKException as ex:
                   print('Failed to refresh flag: {}'.format(ex))
            elif key == ord('c'):
                self._active_mode_index = (self._active_mode_index + 1) % len(self._operation_modes)
                try:
                    self._imager.setActiveOperationMode(self._operation_modes[self._active_mode_index])
                except otc.SDKException as ex:
                    print('Failed to set active operation mode: {}'.format(ex))   

        # Clean up
        cv2.destroyAllWindows()
        self._imager.stopRunning()
          

    # Client callbacks
    def onFrame(self, evt):
        """
        Called when a new thermal frame is available.
        """
        # Since two threads are used synchronization is required
        with self._frame_event_lock:
            # Store a copy of the event to be processed in the main thread. The provided event object is reused
            # by the SDK once the callback returns.
            self._frame_event         = evt.clone()
            self._frame_event_updated = True

            self._fps.trigger()


    def onConnection(self, evt):
        """
        Called when the connection status to the camera changes.
        """
        # Disconnect if the device connection is lost or timed out.
        if evt.state == otc.ConnectionState_Lost or evt.state == otc.ConnectionState_Timeout:
            self._imager.disconnect() 


    def drawOverlay(self, image, fps, flag_state):
        """
        Superimposes an overlay over the false color image.
        """
        mode = self._imager.getActiveOperationMode()

        opticsText = mode.getOpticsText()
        if len(opticsText) != 0:
            opticsText = ' {}'.format(opticsText)

        # Overlay text
        text = ['{}{}, [{:,g}, {:,g}], {}x{} @ {} Hz'.format(mode.getFieldOfView(), opticsText, 
                                                             mode.getTemperatureLowerLimit(), mode.getTemperatureUpperLimit(),
                                                             mode.getFrameWidth(), mode.getFrameHeight(), mode.getFramerate()),
                'FPS: {} (src)'.format(round(fps, 1)), 
                'Flag State: {}'.format(otc.flagStateToString(flag_state)), 
                'q: Quit', 
                'r: Refresh Flag',
                'c: Cycle Operation Modes']

        # Text font face
        thickness = 1
        line_margin = 10

        # Overlay position
        x = image.shape[1] - 275
        y = 25

        # Draw overlay
        for i, line in enumerate(text):
            line_height = cv2.getTextSize(line, self.FONT_FACE, self.FONT_SIZE, thickness)[0][1]
            position    = (x, y + i * (line_height + line_margin))

            cv2.putText(image, line, position, self.FONT_FACE, self.FONT_SIZE, self.BLACK, thickness+1, lineType = cv2.LINE_AA)
            cv2.putText(image, line, position, self.FONT_FACE, self.FONT_SIZE, self.GREEN, thickness, lineType = cv2.LINE_AA)
      

    def calculateAndDrawTemperatureRegions(self, image):
        """
        Determines the hottest and coldest region as well as the mean temperature in a small region in the center and draws
        their position in the frame.
        """
        # Determine the coldest and hottest region with the given radius in the thermal frame
        min_region = otc.TemperatureRegion()
        max_region = otc.TemperatureRegion()
        radius     = 3

        if (self._builder.getMinMaxRegions(radius, min_region, max_region)):
            # Draws a blue crosshair at the center of the lowest temperature region in the thermal frame
            self.drawMeasurement(image, 
                                (int((min_region.x1 + min_region.x2) / 2), int((min_region.y1 + min_region.y2) / 2)),
                                min_region.temperature,
                                self.BLUE,
                                self.WHITE)
            # Draws a red crosshair at the center of the hottest temperature region in the thermal frame
            self.drawMeasurement(image, 
                                (int((max_region.x1 + max_region.x2) / 2), int((max_region.y1 + max_region.y2) / 2)),
                                max_region.temperature,
                                self.RED,
                                self.WHITE)
            
        # Calculate the mean temperature in a small region in the center of thermal frame
        half_image_width  = int(image.shape[1] / 2)
        half_image_height = int(image.shape[0] / 2)
        mean_region = otc.TemperatureRegion(int(half_image_width  - radius),
                                            int(half_image_height - radius),
                                            int(half_image_width  + radius),
                                            int(half_image_height + radius))
        
        # Draws a white cross hair in the center of the display with the mean temperature  
        if (self._builder.getMeanTemperatureInRegion(mean_region)):
            self.drawMeasurement(image,
                                 (int(half_image_width - 1), int(half_image_height - 1)),
                                 mean_region.temperature,
                                 self.WHITE,
                                 self.BLACK)


    def drawMeasurement(self, image, position, value, color, bg_color):
      """
      Draws a marker at the position of a temperature measurement and its value next to it.
      """
      marker_type = cv2.MARKER_CROSS
      marker_size = 20
      thickness   = 1
      
      # Measurement position
      cv2.drawMarker(image, position, bg_color, markerType=marker_type, markerSize=marker_size+1, thickness=thickness+1, line_type=cv2.LINE_AA)
      cv2.drawMarker(image, position, color, markerType=marker_size, markerSize=marker_size, thickness=thickness, line_type=cv2.LINE_AA)

      # Measurement value
      text          = "{}".format(round(value, 1))
      text_position = (position[0] + int(marker_size / 3), position[1] - int(marker_size / 3))
      cv2.putText(image, text, text_position, self.FONT_FACE, self.FONT_SIZE, bg_color, thickness+1, lineType=cv2.LINE_AA)
      cv2.putText(image, text, text_position, self.FONT_FACE, self.FONT_SIZE, color, thickness, lineType=cv2.LINE_AA)
         

def main():
    """
    Main entry point.
    """
    # Get the serial number from command line argument
    # With a serial number of 0 the first compatible camera will be chosen
    serial_number = 0
    if len(sys.argv) >= 2:
        serial_number = int(sys.argv[1])
    
    # Initialize the SDK by providing logger verbosity
    otc.Sdk.init(otc.Verbosity_Info, otc.Verbosity_Off, sys.argv[0])

    # Add an additional detector for Ethernet devices on the network 192.168.0.0/24. A detector for USB devices was added
    # when calling Sdk.init().
    otc.EnumerationManager.getInstance().addEthernetDetector("192.168.0.0/24")
    
    client = None
    try:
        client = ImagerShow(serial_number)
    except otc.SDKException as ex:
        print(ex)
        return

    # Run
    client.run()


if __name__ == "__main__":
    main()