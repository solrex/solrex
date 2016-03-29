
import java.io.*;
import java.io.PrintWriter;
import java.io.IOException;
import java.io.PrintStream;
import com.google.protobuf.Message.Builder;
import com.google.protobuf.TextFormat;
import com.google.protobuf.*;
import org.json.*;
import java.util.*;
import java.nio.*;
import java.util.zip.*;

import com.google.search.app.*;

public class GoogleAppAPI {
	
	public static void decodeS() throws Exception {
		FileInputStream fis = new FileInputStream("s.dat");
		PrintWriter fpw = new PrintWriter("s.txt");
		
		Textsearch.SearchResponse.Builder builder = Textsearch.SearchResponse.newBuilder();
		while (fis.available() > 0) {
			if(true == builder.mergeDelimitedFrom(fis)) {
				Textsearch.SearchResponse search_response = builder.build();
				String title = "#### Proto: Textsearch.SearchResponse with Size=" 
						      + search_response.getSerializedSize();
				fpw.println(title);
				TextFormat.printUnicode(builder, fpw);
				if (search_response.hasSug()) {
					Textsearch.SearchResultPart sug = search_response.getSug();
					String sugstr = sug.getTextData();
					JSONArray jo = new JSONArray(sugstr);
					fpw.println("## JSON: sug_data(JSONArray) ##");
					fpw.println(jo.toString(2));
				}
			}
			builder.clear();
		}
		fis.close();
		fpw.close();
	}
	
	public static void decodeDown() throws Exception {
		FileInputStream fis = new FileInputStream("down.dat");
		PrintWriter fpw = new PrintWriter("down.txt");
		
		// Read Prefix 
		CodedInputStream cis = CodedInputStream.newInstance(fis);
		int version = Integer.reverseBytes(cis.readFixed32());
		fpw.println("version=" + version);
		
		// Read Protobuf block Array
		Voicesearch.VoiceSearchResponse.Builder builder = Voicesearch.VoiceSearchResponse.newBuilder();	
		while (!cis.isAtEnd()) {
			fpw.print("######## Raw Data block info: offset=0x" + Integer.toHexString(cis.getTotalBytesRead()));
			
			int block_size = Integer.reverseBytes(cis.readFixed32());
			fpw.println(" size=" + block_size);
			
			byte[] msgBuf = cis.readRawBytes(block_size);
			builder.mergeFrom(msgBuf);
			Voicesearch.VoiceSearchResponse voice_search_response = builder.build();
			String title = "#### Proto: Voicesearch.VoiceSearchResponse Message with Size=" 
						  + voice_search_response.getSerializedSize();
			fpw.println(title);
			TextFormat.printUnicode(builder, fpw);
			if (voice_search_response.hasSearchResult()
				&& voice_search_response.getSearchResult().hasBodyBytes() ) {
				Textsearch.SearchResponse.Builder text_builder = Textsearch.SearchResponse.newBuilder();
				InputStream bodyByteIS = voice_search_response.getSearchResult().getBodyBytes().newInput();
				while (bodyByteIS.available() > 0) {
					text_builder.mergeDelimitedFrom(bodyByteIS);
					Textsearch.SearchResponse search_response = text_builder.build();
					title = "## Bodybytes Proto: Textsearch.SearchResponse with Size="
					        + search_response.getSerializedSize();
					fpw.println(title);
					TextFormat.printUnicode(text_builder, fpw);
					text_builder.clear();
				}
			}
			fpw.println("\n");
			builder.clear();
		}
		fis.close();
		fpw.close();
	}
	
	public static void decodeUp() throws Exception {
		FileInputStream fis = new FileInputStream("up.dat");
		PrintWriter fpw = new PrintWriter("up.txt");
		
		// Read Prefix
		CodedInputStream cis = CodedInputStream.newInstance(fis);
		cis.readRawBytes(0xa);
		
		Voicesearch.VoiceSearchRequest.Builder builder = Voicesearch.VoiceSearchRequest.newBuilder();	
		// Read Protobuf block Array
		while (!cis.isAtEnd()) {
			int blockOffset = cis.getTotalBytesRead();
			fpw.print("\n######## Data block info: offset=0x" + Integer.toHexString(blockOffset));
			int blockSize = Integer.reverseBytes(cis.readFixed32());
			fpw.println(" blockSize=" + blockSize);
			byte[] msgBuf = cis.readRawBytes(blockSize);
			builder.mergeFrom(msgBuf);
			Voicesearch.VoiceSearchRequest voice_search_request = builder.build();
			String title = "#### Proto: Voicesearch.VoiceSearchRequest Message with Size="
			              + voice_search_request.getSerializedSize();
			fpw.println(title);
			if (voice_search_request.hasGetMethod()) {
				Voicesearch.HttpHeader http_header = voice_search_request.getGetMethod().getHeaders();
				for (int i=0; i<http_header.getNameCount(); i++) {
					if (http_header.getName(i).equalsIgnoreCase("Cookie")) {
                        builder.getGetMethodBuilder().getHeadersBuilder().setTextValue(i, "******");
					}
				}
			}
			if (voice_search_request.hasUserInfo()) {
				builder.getUserInfoBuilder().getGoogleNowBuilder().setAuthKey("******");
				builder.getUserInfoBuilder().setUid("******");
			}
			TextFormat.printUnicode(builder, fpw);
			
			fpw.println("\n");
			builder.clear();
		}
		fis.close();
		fpw.close();
	}
	
	public static void main(String[] args) throws Exception {
		decodeS();
		decodeDown();
		decodeUp();
	}
}
