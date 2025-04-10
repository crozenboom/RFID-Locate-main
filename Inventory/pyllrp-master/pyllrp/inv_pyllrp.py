#!/usr/bin/env python
import time
import socket
import csv
from datetime import datetime
import argparse
try:
    from .pyllrp import *
    from .LLRPConnector import LLRPConnector
except Exception as e:
    from pyllrp import *
    from LLRPConnector import LLRPConnector

class TagInventory:
    roSpecID = 123                  # Arbitrary roSpecID.
    inventoryParameterSpecID = 1234 # Arbitrary inventory parameter spec id.

    def __init__(self, host='192.168.0.219', duration=10, transmitPower=None, receiverSensitivity=None):
        self.host = host
        self.connector = None
        self.duration = duration * 1000  # Convert seconds to milliseconds for LLRP
        self.transmitPower = transmitPower
        self.receiverSensitivity = receiverSensitivity
        self.resetTagInventory()
        # CSV file setup
        self.csv_file = f"tag_inventory_{datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}.csv"
        self.csv_fieldnames = ['Timestamp', 'EPC', 'RSSI', 'AntennaID', 'PhaseAngle', 'ChannelFrequency', 'DopplerFrequency']
        with open(self.csv_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.csv_fieldnames)
            writer.writeheader()

    def resetTagInventory(self):
        self.tagInventory = []
        self.otherMessages = []

    def AccessReportHandler(self, connector, accessReport):
        for tag in accessReport.getTagData():
            epc = HexFormatToStr(tag['EPC'])
            antenna_id = tag.get('AntennaID', 'Unknown')
            rssi = tag.get('PeakRSSI', 'Unknown')
            timestamp = tag.get('FirstSeenTimestampUTC', 'Unknown')
            # Check for Impinj custom parameters
            phase_angle = 'Unknown'
            channel_frequency = 'Unknown'
            doppler_frequency = 'Unknown'
            for param in tag.get('Custom', []):
                if param['VendorID'] == 25882:  # Impinj Vendor ID
                    if param['Subtype'] == 1:  # ImpinjRFPhaseAngle
                        phase_angle = param['Data']
                    elif param['Subtype'] == 2:  # ImpinjRFDopplerFrequency
                        doppler_frequency = param['Data']
                    elif param['Subtype'] == 3:  # ImpinjChannelFrequency
                        channel_frequency = param['Data']

            tag_data = {
                'Timestamp': timestamp,
                'EPC': epc,
                'RSSI': rssi,
                'AntennaID': antenna_id,
                'PhaseAngle': phase_angle,
                'ChannelFrequency': channel_frequency,
                'DopplerFrequency': doppler_frequency
            }
            self.tagInventory.append(tag_data)
            # Print to console
            print(f"Timestamp: {timestamp}, EPC: {epc}, RSSI: {rssi}, Antenna: {antenna_id}, "
                  f"Phase Angle: {phase_angle}, Channel Frequency: {channel_frequency}, Doppler Frequency: {doppler_frequency}")
            # Write to CSV
            with open(self.csv_file, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.csv_fieldnames)
                writer.writerow(tag_data)

    def DefaultHandler(self, connector, message):
        self.otherMessages.append(message)

    def _getReaderConfigMessage(self):
        receiverSensitivityParameter = []
        if self.receiverSensitivity is not None:
            receiverSensitivityParameter.append(
                RFReceiver_Parameter(
                    ReceiverSensitivity=self.receiverSensitivity
                )
            )
        
        transmitPowerParameter = []
        if self.transmitPower is not None:
            transmitPowerParameter.append(
                RFTransmitter_Parameter(
                    TransmitPower=self.transmitPower,
                    HopTableID=1,
                    ChannelIndex=1,
                )
            )
        
        return SET_READER_CONFIG_Message(Parameters=[
            AntennaConfiguration_Parameter(AntennaID=0, Parameters=
                receiverSensitivityParameter +
                transmitPowerParameter + [
                C1G2InventoryCommand_Parameter(Parameters=[
                    C1G2SingulationControl_Parameter(
                        Session=0,
                        TagPopulation=100,
                        TagTransitTime=3000,
                    ),
                    # Enable Impinj extensions in the inventory command
                    Custom_Parameter(
                        VendorID=25882,  # Impinj Vendor ID
                        Subtype=2,       # ImpinjTagReportContentSelector
                        Parameters=[
                            ImpinjTagReportContentSelector_Parameter(
                                EnableRFPhaseAngle=True,
                                EnablePeakRSSI=True,
                                EnableRFDopplerFrequency=True,
                                EnableChannelFrequency=True
                            )
                        ]
                    )
                ]),
            ]),
        ])

    def Connect(self, max_retries=3, retry_delay=2):
        self.connector = LLRPConnector()
        for attempt in range(1, max_retries + 1):
            print(f"Attempting to connect to reader at {self.host}:5084 (Attempt {attempt}/{max_retries})...")
            try:
                # Call connect first to initialize the socket
                response = self.connector.connect(self.host)
                # Now that the socket is created, adjust the timeout
                self.connector.readerSocket.settimeout(15)
                print("Connection successful!")
                break
            except socket.timeout as e:
                print(f"Connection attempt timed out after 6 seconds: {e}")
                if attempt < max_retries:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    self.connector.disconnect()
                    raise
            except socket.error as e:
                print(f"Connection attempt failed: {e}")
                if attempt < max_retries:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    self.connector.disconnect()
                    raise

        # Reset to factory defaults
        response = self.connector.transact(SET_READER_CONFIG_Message(ResetToFactoryDefault=True))
        assert response.success(), f'SET_READER_CONFIG ResetToFactoryDefault fails\n{response}'

        # Enable Impinj extensions
        impinj_enable_extensions = Custom( # type: ignore
            MessageID=0xeded,
            VendorID=25882,  # Impinj Vendor ID
            Subtype=21       # IMPINJ_ENABLE_EXTENSIONS subtype
        )
        response = self.connector.transact(impinj_enable_extensions)
        assert response.success(), f'Failed to enable Impinj extensions\n{response}'

        # Apply reader configuration
        message = self._getReaderConfigMessage()
        response = self.connector.transact(message)
        assert response.success(), f'SET_READER_CONFIG Configuration fails:\n{response}'

    def Disconnect(self):
        self.connector.disconnect()
        self.connector = None

    def GetROSpec(self, antennas=None):
        if antennas is not None:
            if not isinstance(antennas, list):
                antennas = [antennas]
        else:
            antennas = [0]
    
        return ADD_ROSPEC_Message(Parameters=[
            ROSpec_Parameter(
                ROSpecID=self.roSpecID,
                CurrentState=ROSpecState.Disabled,
                Parameters=[
                    ROBoundarySpec_Parameter(
                        Parameters=[
                            ROSpecStartTrigger_Parameter(ROSpecStartTriggerType=ROSpecStartTriggerType.Immediate),
                            ROSpecStopTrigger_Parameter(
                                ROSpecStopTriggerType=ROSpecStopTriggerType.Duration,
                                Duration=self.duration
                            ),
                        ]
                    ),
                    AISpec_Parameter(
                        AntennaIDs=antennas,
                        Parameters=[
                            AISpecStopTrigger_Parameter(
                                AISpecStopTriggerType=AISpecStopTriggerType.Null,
                            ),
                            InventoryParameterSpec_Parameter(
                                InventoryParameterSpecID=self.inventoryParameterSpecID,
                                ProtocolID=AirProtocols.EPCGlobalClass1Gen2,
                            ),
                        ]
                    ),
                    ROReportSpec_Parameter(
                        ROReportTrigger=ROReportTriggerType.Upon_N_Tags_Or_End_Of_AISpec,
                        N=1,  # Report every tag read immediately
                        Parameters=[
                            TagReportContentSelector_Parameter(
                                EnableAntennaID=True,
                                EnableFirstSeenTimestamp=True,
                                EnablePeakRSSI=True,
                            ),
                        ]
                    ),
                ]
            ),
        ])

    def _prolog(self, antennas=None):
        response = self.connector.transact(DISABLE_ROSPEC_Message(ROSpecID=0))
        response = self.connector.transact(DELETE_ROSPEC_Message(ROSpecID=self.roSpecID))

        self.resetTagInventory()
        self.connector.addHandler(RO_ACCESS_REPORT_Message, self.AccessReportHandler)
        self.connector.addHandler('default', self.DefaultHandler)

        response = self.connector.transact(self.GetROSpec(antennas))
        assert response.success(), f'Add ROSpec Fails\n{response}'

    def _execute(self):
        response = self.connector.transact(ENABLE_ROSPEC_Message(ROSpecID=self.roSpecID))
        assert response.success(), f'Enable ROSpec Fails\n{response}'

        # Wait for the duration plus a small buffer
        time.sleep((self.duration / 1000.0) + 1.0)

        response = self.connector.transact(DISABLE_ROSPEC_Message(ROSpecID=self.roSpecID))
        assert response.success(), f'Disable ROSpec Fails\n{response}'

    def _epilog(self):
        response = self.connector.transact(DELETE_ROSPEC_Message(ROSpecID=self.roSpecID))
        assert response.success(), f'Delete ROSpec Fails\n{response}'
        self.connector.removeAllHandlers()

    def GetTagInventory(self, antennas=None):
        self._prolog(antennas)
        self._execute()
        self._epilog()
        return self.tagInventory, self.otherMessages

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='RFID Tag Inventory Script')
    parser.add_argument('--power', type=int, default=91, help='Transmit power level (default: 91)')
    parser.add_argument('--duration', type=int, default=10, help='Inventory duration in seconds (default: 10)')
    args = parser.parse_args()

    print(f"Running inventory at power level {args.power} for {args.duration} seconds")
    host = '192.168.0.219'
    ti = TagInventory(host, duration=args.duration, transmitPower=args.power)
    ti.Connect()
    tagInventory, otherMessages = ti.GetTagInventory()
    ti.Disconnect()