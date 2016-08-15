import javax.crypto.Cipher;
import javax.crypto.spec.IvParameterSpec;
import javax.crypto.spec.SecretKeySpec;
import javax.crypto.CipherInputStream;

import java.io.FileInputStream;
import java.io.InputStream;

public class decrypt {

    private static byte[] hexToBytes(String s)
    {
        int len = s.length();
        byte[] data = new byte[len / 2];

        for (int i = 0; i < len; i += 2)
            data[i / 2] = (byte) ((Character.digit(s.charAt(i), 16) << 4) + Character.digit(s.charAt(i + 1), 16));

        return data;
    }

    public static void main(String[] args) throws Exception {
        InputStream cipherInputStream = null;
        try {
            final StringBuilder output = new StringBuilder();
            final Cipher cipher = Cipher.getInstance("AES/CBC/NoPadding");
            cipher.init(Cipher.DECRYPT_MODE, new SecretKeySpec(hexToBytes("25c506a9e4a0b3100d2d86b49b83cf9a"), 0, 16, "AES"),
                    new IvParameterSpec(hexToBytes("00000000000000000000000000000000")));
            cipherInputStream = new CipherInputStream(new FileInputStream("data.enc"), cipher);

            final String charsetName = "UTF-8";

            final byte[] buffer = new byte[8192];
            int read = cipherInputStream.read(buffer);

            while (read > -1) {
                output.append(new String(buffer, 0, read, charsetName));
                read = cipherInputStream.read(buffer);
            }

            System.out.println(output);
        } finally {
            if (cipherInputStream != null) {
                cipherInputStream.close();
            }
        }
    }
}
