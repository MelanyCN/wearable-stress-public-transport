import io, contextlib, warnings; warnings.filterwarnings('ignore')
import h1_ruido_fc as H
for nombre,(db,q,sep) in {'primaria (6 dB, top10%, sep30)':(6,.9,30),'sens-1 (4 dB, top20%, sep20)':(4,.8,20),'sens-2 (3 dB, top30%, sep20)':(3,.7,20)}.items():
    H.UMBRAL_DB,H.QUANT,H.SEP=db,q,sep
    buf=io.StringIO()
    with contextlib.redirect_stdout(buf): H.main()
    out=buf.getvalue().splitlines()
    print('==',nombre); [print('  ',l) for l in out if l.startswith(('H1','Ajustado','Dosis','Por participante'))]
