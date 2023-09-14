#include <Bela.h>
#include <stdlib.h> //access to console
#include <cmath>
#include <libraries/math_neon/math_neon.h> //ARM Cortex A8 optimised math

//Define global variables and arrays
//Button pins and states
const int ButPin[5] = { 14, 13, 12, 8, 9 };
int ButState[5] = { 0, 0, 0, 0, 0 };

//LED pins and states
const int LedPin[5] = { 4, 2, 3, 0, 1 };
int LedState[5] = { 0, 0, 0, 0, 0 };

//Pot pins and states (analogue in)
const int PotPin[3] = { 5, 7, 6 };
float PotState[3] = { 0.0f, 0.0f, 0.0f };

//Button counters
long B1count = 0, B2count = 0, B3count = 0, B5count = 0;

//Single press and 5s hold length, frames
const int gSinglePress = 3000;
const long g5sHold = 150000;

//B3 counter freeze timer
long B3ctimer = 0;

//B5 press logged time
long B5pressed;

//Play noise variable
int gPlay = 0;

//Shutdown trigger variable
int gShutdown = 0;

//Block time variable and frequenctly called block time equivalents (timestamps, FLOORs)
long gTime, g10min, g11min, g12min;

//Sample rate, block size
long gSamplerate;
int gBlockSize;

//Constant to convert [s] to block time
float gBlockTimeConvert;

//Constant to convert frame time to [s]
float gSample2s;

//Phase of oscillator
float gPhase = 0.0f;


//Define functions
//Real-time sine wave generator, 50-1k Hz [samples, 0-1, 0-1], output halved
//Only works as phase master
float sine(float frequency, float amplitude) {
	frequency = frequency * 950.0f + 50.0f; //Rescale potmeter value to 50-950
	gPhase += 2.0f * (float)M_PI * frequency * gSample2s; //Increase phase
	if (gPhase > 1.0f) //Constrain phase in a ~2pi range
	{
		gPhase -= 2.0f * (float)M_PI;
	}
	return amplitude * sinf_neon(gPhase);
}

//Real-time white noise generator [0-1], output halved
float whitenoise(float amplitude) {
	return amplitude * (2.0f * (float)rand() / (float)RAND_MAX - 1.0f) / 2.0f; //Explained in dissertation
}

//Real-time even pulse with phase delay [Hz, rad, 0/1]
//Set master to 0 to control phase (not delay!!) externally
int pulse(float frequency, float phasedelay, int master) {
	gPhase += (float)master * 2.0f * (float)M_PI * frequency * gSample2s; //Increase phase
	if (gPhase > 1.0f) //Constrain phase in a ~2pi range
	{
		gPhase -= (float)master * 2.0f * (float)M_PI;
	}
	return round((sinf_neon(gPhase + phasedelay) + 1.0f) / 2.0f); //Explained in dissertation
}

//Setup
bool setup(BelaContext* context, void* userData)
{
	// Check that audio and digital have the same number of frames
	if (context->audioFrames != context->digitalFrames) {
		rt_fprintf(stderr, "Please match analogue and digital sample rate settings.\n");
		return false;
	}

	//Check that analogue sample rate is half of digital
	if (context->analogFrames != context->digitalFrames / 2) {
		rt_fprintf(stderr, "Please set analogue sample rate to half of audio sample rate.\n");
		return false;
	}

	//Set the mode of digital pins
	for (unsigned int i = 0; i < 5; i++)
	{
		pinMode(context, 0, ButPin[i], INPUT);
		pinMode(context, 0, LedPin[i], OUTPUT);
	}

	//Get audio block size and sample rate
	//Calculate conversion constants
	gSamplerate = context->audioSampleRate;
	gBlockSize = context->audioFrames;
	gBlockTimeConvert = (float)gSamplerate / (float)gBlockSize;
	gSample2s = 1.0f / (float)gSamplerate;

	//Set time to -16s for startup
	gTime = -16.0f * gBlockTimeConvert;

	//Calculate timestamp FLOORs
	g10min = floorf(10.0f * 60.0f * gBlockTimeConvert);
	g11min = floorf(11.0f * 60.0f * gBlockTimeConvert);
	g12min = floorf(12.0f * 60.0f * gBlockTimeConvert);

	return true;
}

//Render
void render(BelaContext* context, void* userData)
{
	if (gTime < 0)
	{
		if (gTime < -10.0f * gBlockTimeConvert) //Startup
		{
			//LED startup sequence, 5s window, 60 bpm
			LedState[0] = 1;
			for (unsigned int n = 0; n < gBlockSize; n++)
			{
				LedState[1] = pulse(0.125, -(float)M_PI / 4.0f, 1);
				LedState[2] = pulse(0, -(float)M_PI / 2.0f, 0);
				LedState[3] = pulse(0, -3.0f * (float)M_PI / 4.0f, 0);
				LedState[4] = pulse(0, -M_PI, 0);

				for (unsigned int i = 0; i < 5; i++)
				{
					digitalWriteOnce(context, n, LedPin[i], LedState[i]);
				}
			}

			if (gTime > -11.0f * gBlockTimeConvert) //Exit startup
			{
				for (unsigned int i = 0; i < 5; i++) //Turn off LEDs
				{
					digitalWrite(context, 0, LedPin[i], 0);
				}
				gTime = -1;
			}
		}
		else //Freeze inputs/outputs while wifi is connecting
		{
			//Flash LED1
			for (unsigned int n = 0; n < gBlockSize; n++)
			{
				LedState[0] = pulse(2, 0, 1);
				digitalWriteOnce(context, n, LedPin[0], LedState[0]);
			}
		}
	}
	else if (gTime < g10min) //Listen to inputs and write to outputs
	{
		for (unsigned int n = 0; n < gBlockSize; n++) //Frame-level
		{
			for (unsigned int i = 0; i < 5; i++) //Read and save button states
			{
				ButState[i] = digitalRead(context, n, ButPin[i]);
			}
			for (unsigned int i = 0; i < 3; i++) //Read and save potentiometer states
			{
				PotState[i] = analogRead(context, n / 2, PotPin[i]) / 0.82f; //Scaled to 0-1 (3.3V max, not 4V)
			}

			//Buttons
			if (ButState[0] > 0) //Increase B1 counter, illuminate while pressed
			{
				B1count++;
				digitalWriteOnce(context, n, LedPin[0], 1);
				digitalWriteOnce(context, gBlockSize - 1, LedPin[0], 0); //Make sure LEDs are turned off after B4 is pressed
			}
			if (ButState[1] > 0) //Increase B2 counter
			{
				B2count++;
			}
			if ((ButState[2] > 0) && (B3ctimer > 0.2f * gBlockTimeConvert)) //If button pressed over 0.2s ago...
			{
				B3count++; //...increase B3 counter
			}
			if (ButState[3] > 0) //Illuminate LEDs when B4 is pressed
			{
				for (unsigned int i = 0; i < 5; i++)
				{
					digitalWriteOnce(context, n, LedPin[i], 1);
					digitalWriteOnce(context, gBlockSize - 1, LedPin[i], 0); //Make sure LEDs are turned off after B4 is pressed
				}
			}


			if (ButState[4] > 0) //Increase B5 counter
			{
				B5count++;
			}

			//Auduio output
			//float out = gPlay * (sine(gPhaseTime, PotState[0], PotState[1]) + whitenoise(PotState[2]));
			float out = (float)gPlay * (sine(PotState[0], PotState[1]) + whitenoise(PotState[2]));
			audioWrite(context, n, 0, out); //Left channel
			audioWrite(context, n, 1, 0.0f); //Right channel, muted
		}

		//Block-level
		if (B1count > gSinglePress) //B1 action trigger
		{
			B1count = 0; //Reset button counter
			gPhase = 0.0f; //Set zero phase for 'connect' LED seq.
			gTime = ceilf(-3.0f * gBlockTimeConvert); //Enter freeze
			system("ifdown wlan0; ifup wlan0;"); //Wifi reconnect
		}
		if (B2count > gSinglePress) //B2 action trigger
		{
			gPhase = 0.0f; //Set zero phase for 'next' LED seq.
			gTime = g10min; //Proceed to 'next' LED seq.
		}
		if (B3count > gSinglePress) //B3 action trigger
		{
			gPlay = (gPlay + 1 == 2) ? 0 : 1; //Cycle through 0-1
			B3count = 0; //Reset button counter
			B3ctimer = 0; //Reset counter timer
		}
		B3ctimer++; //Increase B3 counter timer

		if (B5count > g5sHold) //B5 action trigger
		{
			gPhase = 0.0f; //Set zero phase for shutdown LED seq.
			gTime = g11min; //Proceed to shutdown LED sequence
		}
		else if (B5count > 100) //B5 counter reset trigger
		{
			if (B5count < 117) //Log time pressed
			{
				B5pressed = gTime;
			}
			else if (gTime - B5pressed > 8.0f * gBlockTimeConvert) //Reset counter after 8 secs if pressed for <5s
			{
				B5count = 0;
			}
		}
	}
	else if (gTime < g11min) //'Next' LED sequence
	{
		//Flash LED1-5 4 times (100bpm, 2.5s window)
		for (unsigned int n = 0; n < gBlockSize; n++)
		{
			LedState[0] = pulse(1.67, -0.7, 1);

			for (unsigned int i = 0; i < 5; i++)
			{
				digitalWriteOnce(context, n, LedPin[i], LedState[0]);
			}
		}

		if (gTime > g10min + 2.5f * gBlockTimeConvert) //Poceed to stop project
		{
			gTime = g12min;
		}
	}
	else if (gTime < g12min) //Shutdown LED sequence
	{
		gShutdown = 1; //Initiate shutdown in cleanup

		//LED flash pattern: all>2+3+4>3>none (~80bpm, 2.5s window)
		LedState[2] = 1;
		for (unsigned int n = 0; n < gBlockSize; n++)
		{
			LedState[0] = LedState[4] = pulse(0.286, 1.35, 1);
			LedState[1] = LedState[3] = pulse(0, 0, 0);

			for (unsigned int i = 0; i < 5; i++)
			{
				digitalWriteOnce(context, n, LedPin[i], LedState[i]);
			}
		}

		if (gTime > g11min + 2.5f * gBlockTimeConvert) //Poceed to stop project
		{
			gTime = g12min;
		}
	}
	else //Stop project
	{
		for (unsigned int i = 0; i < 5; i++) //Turn off LEDs
		{
			digitalWrite(context, 0, LedPin[i], 0);
		}

		Bela_requestStop();
	}

	gTime++; //Increase block time
}

//Cleanup
void cleanup(BelaContext* context, void* userData)
{
	if (gShutdown == 1) //Shutdown if selected or timeout
	{
		system("/root/Bela/scripts/halt_board.sh");
	}
}